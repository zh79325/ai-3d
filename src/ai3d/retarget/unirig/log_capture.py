"""把 UniRig 执行期间的**全部输出**落到素材目录的 ``unirig.log``。

UniRig 是进程内调用（见 :mod:`predict`），没有子进程 stdout 管道可收：它内部的
Lightning 进度条、``print`` 与 logging 全打在服务进程的 ``sys.stdout`` / ``sys.stderr``
上，和 HTTP 请求日志混成一团，前端只能看到 ``bindings.stage`` 那几个粗粒度状态，
一次推理好几分钟里完全不知道卡在哪。

本模块在绑定期间把这三个来源**按线程**分流到日志文件：

- 注册线程（绑定线程 + 它派生的推理线程）的输出：既写原流（服务控制台不丢东西），
  也写日志文件；
- 其它线程（FastAPI 请求处理、别的素材同时在跑的 S2）：只写原流，绝不串进来。

按线程分流是必需的 —— ``sys.stdout`` 是进程级单例，而 S2 的锁是 **per-asset** 的，
两个素材完全可能同时在跑 UniRig；不做线程过滤两份日志会互相污染。

两个实现约束：

1. **写入即 flush**。前端按字节 offset 增量轮询这个文件，缓冲不落盘就等于「进度卡住」；
2. **``\\r`` 与 ``\\n`` 一样视作行结束**。tqdm 靠 ``\\r`` 原地刷新进度条，只认 ``\\n``
   的话整个推理过程一行都出不来。

还有个反直觉的坑：**最耗时的一段恰恰没有任何输出**。骨架阶段是 HF ``generate``
的单 batch 自回归（``max_new_tokens=2048``、``num_beams=15``），Lightning 的进度条
只覆盖「1 个 batch」，实测 ``+26s`` 到 ``+226s`` 整整 200s 静默 —— 光把输出接过来
用户依然分不清是在算还是卡死了。故这里补一个**心跳**：静默超过 ``HEARTBEAT_S``
就自己写一行「已运行 / 已静默」，配合 :func:`set_phase` 打的阶段标签，静默期也能
看出「跑到哪了、还活着」。

用法::

    with capture(out_dir / "unirig.log", header="==== RUN ... ===="):
        predict_unirig(...)          # 内部起线程处调 attach_thread()
"""

from __future__ import annotations

import contextlib
import logging
import re
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TextIO

# tqdm 每个 batch 刷一行，几分钟的推理能刷出几万行；写爆了前端也没法看，故设硬上限
MAX_LOG_BYTES = 8 * 1024 * 1024
# 静默多久补一行心跳。取 20s：200s 的骨架阶段正好十来行，既看得出「还在动」，
# 又不会把真实输出淹掉；传 0 给 capture() 可关掉。
HEARTBEAT_S = 20.0
_HEARTBEAT_TICK = 1.0                   # 心跳线程的巡检间隔

# ANSI 转义（tqdm/rich 的光标移动与配色）在网页里是乱码，落盘前剥掉
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[()][A-Z0-9]")
# 行切分：\r\n 先归一成 \n，再按 \r / \n 切
_EOL_RE = re.compile(r"[\r\n]")

_lock = threading.RLock()
# key 用 Thread 对象而不是 get_ident()：ident 是地址级的标识，子线程一结束就会被
# 后续新线程（比如 FastAPI 的工作线程）复用，届时那些请求的输出会凭空串进日志。
# Thread 对象唯一且支持弱引用，线程被回收后条目自动消失。
_writers: "weakref.WeakKeyDictionary[threading.Thread, _Writer]" = (
    weakref.WeakKeyDictionary())
_tee_depth = 0                          # 已装了几层 Tee（归零才还原 sys 流）
_saved_streams: List[TextIO] = []       # 装 Tee 前的 (stdout, stderr)
_handler: Optional["_RegistryHandler"] = None
_heartbeat_thread: Optional[threading.Thread] = None
# 心跳线程的睡眠用 Event 而不是 time.sleep：新 capture 开始时能立即把它叫醒。
# 否则它会抱着上一轮的巡检间隔睡完（默认 1s），刚起步的那一拍心跳就丢了。
_heartbeat_wake = threading.Event()


class _Writer:
    """一份日志文件的写入端：行切分 + 相对时间戳前缀 + 体积上限。

    相对时间戳（``[+  12.3s]``）比绝对时间有用：一眼看出哪一步吃了多少时间，
    这正是排查「骨架预测慢还是蒙皮慢」时最需要的信息。
    """

    def __init__(self, path: Path, heartbeat_s: float = HEARTBEAT_S):
        self.path = Path(path)
        self.t0 = time.time()
        self.owner_threads: List[threading.Thread] = []   # 归属本 writer，退出时统一摘掉
        # 推理线程、logging handler、心跳线程会同时写同一份文件，``_buf`` 是共享
        # 状态：不加锁两行输出会互相截断（半行丢失）。
        self._wlock = threading.RLock()
        self._buf = ""
        self._bytes = 0
        self._closed = False
        self._broken = False                    # 写盘失败过（磁盘满等），此后不再试
        self._noted_overflow = False
        self._last_emit = self.t0               # 最后一次落盘时刻，心跳据此算静默时长
        self.heartbeat_s = float(heartbeat_s)
        self.phase: Optional[str] = None        # 当前阶段标签（:func:`set_phase` 写入）
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 覆盖式：这份日志只代表「本次运行」。上一次的失败原因已落 bindings.error，
        # 追加模式会让前端在增量读取时分不清哪段属于哪一轮。
        self._fh = self.path.open("w", encoding="utf-8", errors="replace", newline="\n")

    def _write_raw(self, text: str) -> None:
        """落盘；写不动就永久静默（日志是辅助，绝不能把推理带崩）。"""
        with self._wlock:
            if self._broken:
                return
            try:
                self._fh.write(text)
                self._fh.flush()                # 前端要实时读，不能等缓冲区攒满
                self._last_emit = time.time()
            except (OSError, ValueError) as exc:
                self._broken = True
                # 绕开 Tee 直写原始 stderr：此时 sys.stderr 可能正指回本 writer，会递归
                print(f"[log_capture] 写 {self.path} 失败，后续输出不再落盘：{exc}",
                      file=getattr(sys, "__stderr__", None))

    # ---- 原样落盘（header / footer，不加时间戳前缀） ---- #
    def raw(self, text: str) -> None:
        if self._closed or not text:
            return
        self._write_raw(text if text.endswith("\n") else text + "\n")

    # ---- 带行切分的写入（stdout / stderr 用：调用方的换行不可控） ---- #
    def write(self, text: str) -> None:
        if self._closed or not text:
            return
        with self._wlock:
            self._buf += text.replace("\r\n", "\n")
            parts = _EOL_RE.split(self._buf)
            self._buf = parts.pop()             # 末段可能是不完整行，留到下次
            for line in parts:
                self._emit(line)

    # ---- 整行写入（logging 用：handler.format() 不带末尾换行） ---- #
    def line(self, text: str) -> None:
        """把 ``text`` 当**完整行**写入（可含多行，如 traceback）。

        不能直接用 :meth:`write`：logging 的记录不带末尾换行，丢进 ``_buf`` 后会一直
        留着，下一条 stdout 半行接上去就粘成一行（实测过：``...boom`` 与
        ``from child thread`` 粘连）。故先把残留半行单独落一行，再逐行写本条。
        """
        if self._closed or not text:
            return
        with self._wlock:
            if self._buf:
                self._emit(self._buf)
                self._buf = ""
            parts = _EOL_RE.split(text.replace("\r\n", "\n"))
            if parts and parts[-1] == "":        # 以换行结尾时 split 会多出一个空串
                parts.pop()
            for part in parts:
                self._emit(part)

    def _emit(self, line: str) -> None:
        if self._bytes >= MAX_LOG_BYTES:
            return
        line = _ANSI_RE.sub("", line).rstrip()
        stamp = f"[+{time.time() - self.t0:7.1f}s] "
        self._write_raw(stamp + line + "\n")
        self._bytes += len(stamp) + len(line) + 1
        if self._bytes >= MAX_LOG_BYTES and not self._noted_overflow:
            self._noted_overflow = True
            self.raw(f"…… 输出超过 {MAX_LOG_BYTES // 1024 // 1024}MB，后续不再记录")

    def beat(self) -> None:
        """静默超过阈值就补一行心跳（由心跳线程调，不在推理线程上）。

        写心跳会刷新 ``_last_emit``，故静默期里恰好每 ``heartbeat_s`` 一行，不会刷屏。
        """
        if self._closed or self.heartbeat_s <= 0:
            return
        with self._wlock:
            idle = time.time() - self._last_emit
            if idle < self.heartbeat_s:
                return
        where = f"{self.phase} " if self.phase else ""
        self.line(f"…… {where}已运行 {time.time() - self.t0:.0f}s、"
                  f"静默 {idle:.0f}s（仍在计算，此阶段本就不输出）")

    def close(self) -> None:
        with self._wlock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._buf.strip():           # 收尾：没带换行的最后一段也要落盘
                    self._emit(self._buf)
                self._buf = ""
                self._fh.close()
            except (OSError, ValueError):
                pass


class _TeeStream:
    """替换 ``sys.stdout`` / ``sys.stderr``：注册线程双写，其余线程只写原流。"""

    def __init__(self, original: TextIO):
        self._original = original

    def write(self, s):
        original = self.__dict__.get("_original")
        if original is not None:
            original.write(s)
        writer = _writers.get(threading.current_thread())
        if writer is not None and isinstance(s, str):
            try:
                writer.write(s)
            except (OSError, ValueError):       # 日志写不动不能把推理带崩
                pass
        return len(s)

    def flush(self) -> None:
        original = self.__dict__.get("_original")
        if original is not None:
            original.flush()

    def isatty(self) -> bool:
        # 透传真值：Lightning/tqdm 据此决定是否输出进度条，谎报 tty 会换来一堆
        # 光标控制码，谎报非 tty 又可能让上游干脆不打进度
        original = self.__dict__.get("_original")
        return bool(original.isatty()) if original is not None else False

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8") or "utf-8"

    def __getattr__(self, name):
        # fileno / buffer / errors 等一律透传，否则 uvicorn、tqdm 会 AttributeError
        original = self.__dict__.get("_original")
        if original is None:
            raise AttributeError(name)
        return getattr(original, name)


class _RegistryHandler(logging.Handler):
    """把注册线程产生的日志记录写进它自己那份日志。

    按 ``threading.current_thread()`` 而不是 ``record.thread``（ident）分派：emit 与
    产生记录的调用在同一线程同步发生，而 ident 会被后续线程复用。

    装在 root logger 上、常驻不摘：靠 registry 查不到就 return，开销可忽略，
    比每次 capture 都 addHandler/removeHandler 更不容易在异常路径上漏摘。
    """

    def emit(self, record: logging.LogRecord) -> None:
        writer = _writers.get(threading.current_thread())
        if writer is None:
            return
        try:
            writer.line(self.format(record))
        except (OSError, ValueError):
            self.handleError(record)


def _install_handler() -> None:
    """把日志 handler 挂到 root，并统一格式（只在首次 capture 时做一次）。"""
    global _handler
    root = logging.getLogger()
    if _handler is not None:
        return
    handler = _RegistryHandler(level=logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    _handler = handler


def _ensure_heartbeat() -> None:
    """起一个常驻守护线程给静默的 writer 补心跳（只起一次）。

    不让它“没活就退出”：退出与新 capture 注册之间有竞态窗口，会漏掉开头几拍。
    空闲时它只是一秒一次空字典遍历，开销可忽。
    """
    global _heartbeat_thread
    with _lock:
        if _heartbeat_thread is None or not _heartbeat_thread.is_alive():
            thread = threading.Thread(target=_heartbeat_loop,
                                      name="unirig-log-heartbeat", daemon=True)
            _heartbeat_thread = thread
            thread.start()
        _heartbeat_wake.set()               # 已存在也叫醒：按本轮的间隔重新计时


def _heartbeat_loop() -> None:
    while True:
        _heartbeat_wake.wait(_HEARTBEAT_TICK)
        _heartbeat_wake.clear()
        with _lock:
            # 快照去重：同一 writer 可能被多个线程登记；WeakKeyDictionary 遍历中
            # 条目被回收会报 RuntimeError，故全程持锁
            writers = list({id(w): w for w in _writers.values()}.values())
        for writer in writers:
            try:
                writer.beat()
            except Exception:                   # noqa: BLE001 心跳绝不能反噬推理
                pass


def attach_thread(owner: Optional[threading.Thread] = None) -> None:
    """把**当前线程**的输出接到 ``owner`` 那次 capture（缺省接唯一在跑的那次）。

    :func:`predict._run_with_timeout` 会在守护线程里跑推理（为了超时能放弃等待），
    Lightning 的进度条正是从**这个**线程写 stderr 的 —— 不 attach 等于 unirig.log
    里什么推理输出都没有。线程干完活应调 :func:`detach_thread` 即时摘掉（弱引用
    要等对象被回收才生效，而 ident 可能立刻被复用）。
    """
    current = threading.current_thread()
    with _lock:
        writer = _writers.get(owner) if owner is not None else None
        if writer is None and owner is None:
            distinct = {id(w) for w in _writers.values()}
            if len(distinct) == 1:
                writer = next(iter(_writers.values()))
        if writer is None or _writers.get(current) is writer:
            return
        _writers[current] = writer
        writer.owner_threads.append(current)


def detach_thread() -> None:
    """摘掉当前线程的登记（子线程结束时调，避免它的 ident/对象被后续线程沾用）。"""
    current = threading.current_thread()
    with _lock:
        writer = _writers.pop(current, None)
        if writer is not None and current in writer.owner_threads:
            writer.owner_threads.remove(current)


def set_phase(text: Optional[str]) -> None:
    """给当前 capture 打一个阶段标签，让**心跳行**能说出“卡在哪一步”。

    心跳线程自己不知道业务阶段（它属于 log_capture，不该反过来依赖 predict），
    故由推理侧在切阶段时主动告知；不告知也能跑，只是心跳里少个前缀。
    """
    writer = _writers.get(threading.current_thread())
    if writer is not None:
        writer.phase = text


def _detach(writer: _Writer) -> None:
    """摘掉归属该 writer 的全部线程登记（含超时后仍在跑的僵尸线程）。"""
    with _lock:
        for thread in list(writer.owner_threads):
            if _writers.get(thread) is writer:
                _writers.pop(thread, None)
        writer.owner_threads.clear()


@contextlib.contextmanager
def capture(path: Path | str, header: Optional[str] = None,
            heartbeat_s: float = HEARTBEAT_S) -> Iterator[Path]:
    """捕获期间的 stdout / stderr / logging 全部落 ``path``。

    ``sys`` 流是进程级单例，故用引用计数：并发的第二次 capture 不再套一层 Tee
    （共用同一对即可，反正按线程分派），退出时也要等所有 capture 都结束才还原。

    ``heartbeat_s`` 为静默多久补一行心跳（0 = 关）：推理最耗时的那一段往往一行
    输出都没有，没心跳的话前端会看着一个不动的日志分不清是在算还是挂了。
    """
    global _tee_depth
    path = Path(path)
    _install_handler()
    writer = _Writer(path, heartbeat_s=heartbeat_s)
    current = threading.current_thread()
    root = logging.getLogger()
    saved_level = root.level
    # 注册与还原全部进 try/finally：中途报错也不能把 sys.stdout 留在 Tee 上、
    # 更不能在 registry 里漏下僵尸条目（否则后续所有请求的输出都会往这份日志里灌）
    try:
        with _lock:
            _writers[current] = writer
            writer.owner_threads.append(current)
            if _tee_depth == 0:
                _saved_streams[:] = [sys.stdout, sys.stderr]
                sys.stdout = _TeeStream(sys.stdout)      # type: ignore[assignment]
                sys.stderr = _TeeStream(sys.stderr)      # type: ignore[assignment]
            _tee_depth += 1
        _ensure_heartbeat()
        # UniRig / Lightning 的关键信息多在 INFO；服务默认 WARNING 会让它们全被过滤掉。
        # 只降门槛不动其它 handler 的行为，退出时原样恢复。
        if saved_level > logging.INFO or saved_level == logging.NOTSET:
            root.setLevel(logging.INFO)
        if header:
            writer.raw(header)
        yield path
    finally:
        writer.raw(f"==== 输出结束（共 {time.time() - writer.t0:.1f}s，"
                   f"{writer._bytes} 字节）====")
        with _lock:
            _tee_depth = max(0, _tee_depth - 1)
            if _tee_depth == 0 and _saved_streams:
                sys.stdout, sys.stderr = _saved_streams[0], _saved_streams[1]
                _saved_streams.clear()
        root.setLevel(saved_level)
        _detach(writer)
        writer.close()


def note(path: Path | str, text: str) -> None:
    """往已存在的日志尾部追加一行（capture 之外用，比如失败回滚时补记原因）。"""
    try:
        p = Path(path)
        if not p.exists():
            return
        with p.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write((text if text.endswith("\n") else text + "\n"))
    except OSError:
        pass


__all__ = ["capture", "attach_thread", "detach_thread", "set_phase", "note",
           "MAX_LOG_BYTES", "HEARTBEAT_S"]
