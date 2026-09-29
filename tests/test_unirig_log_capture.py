"""``unirig.log_capture`` 的定向验证。

只测这次新增的输出捕获：捕获来源是否齐（stdout / stderr / ``\\r`` 进度条 / logging /
traceback）、**线程隔离**是否可靠（未登记线程与并发 capture 都不能串线）、退出时
是否把 ``sys.stdout`` 与 root level 原样还原。不碰真实推理（那要几分钟 + 6GB 内存）。
"""

import logging
import sys
import threading
import time

import pytest

from ai3d.retarget.unirig import log_capture as lc

LOG = logging.getLogger("ai3d.test.capture")


@pytest.fixture
def quiet_root():
    """把 root 压到 WARNING（服务的常见状态）：capture 期间应仍能收到 INFO。"""
    root = logging.getLogger()
    saved = root.level
    root.setLevel(logging.WARNING)
    try:
        yield root
    finally:
        root.setLevel(saved)


def test_collects_stdout_stderr_and_carriage_return_progress(tmp_path):
    """print / stderr / tqdm 的 ``\\r`` 刷新都要落盘，且 ANSI 转义被剥掉。"""
    log_file = tmp_path / "unirig.log"
    with lc.capture(log_file, header="==== RUN ===="):
        print("stdout 行")
        sys.stderr.write("stderr 行\n")
        # tqdm 风格：ANSI 配色 + \r 原地刷新，整段不带一个 \n
        sys.stdout.write("Predicting: 10%|\x1b[0m 1/10\r")
        sys.stdout.write("Predicting: 100%| 10/10\r")
        sys.stdout.write("没有换行的尾巴")

    text = log_file.read_text(encoding="utf-8")
    assert text.startswith("==== RUN ====")
    assert "stdout 行" in text
    assert "stderr 行" in text
    # \r 也算行结束：只认 \n 的话整个推理过程一行进度都出不来
    assert text.count("Predicting:") == 2
    assert "\x1b[" not in text
    assert "没有换行的尾巴" in text          # 收尾时残留缓冲也要落盘
    assert "输出结束" in text                 # footer


def test_collects_info_log_and_traceback(tmp_path, quiet_root):
    """root 是 WARNING 时也要收到 INFO；失败 traceback 要能在日志里看到原因。"""
    log_file = tmp_path / "unirig.log"
    with lc.capture(log_file):
        LOG.info("INFO 级别的关键信息")
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            LOG.exception("推理失败")

    text = log_file.read_text(encoding="utf-8")
    assert "INFO 级别的关键信息" in text
    assert "RuntimeError: boom" in text
    assert quiet_root.level == logging.WARNING        # 退出后原样还原


def test_log_record_does_not_glue_onto_half_line(tmp_path):
    """logging 记录不带末尾换行：不能与上一条 stdout 半行粘成一行。"""
    log_file = tmp_path / "unirig.log"
    with lc.capture(log_file):
        sys.stdout.write("没有换行的前半段")     # 故意留在行缓冲里
        LOG.info("紧跟着的一条日志")

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert any("没有换行的前半段" in ln and "紧跟着的一条日志" not in ln for ln in lines)
    assert any("紧跟着的一条日志" in ln for ln in lines)


def test_only_attached_threads_are_captured(tmp_path):
    """attach 过的子线程进日志；未登记的线程（HTTP 请求线程）绝不进。"""
    log_file = tmp_path / "unirig.log"
    with lc.capture(log_file):
        owner = threading.current_thread()

        def attached_child():
            lc.attach_thread(owner)
            try:
                print("已登记的推理线程")
            finally:
                lc.detach_thread()

        t = threading.Thread(target=attached_child)
        t.start()
        t.join(5)

        # 连续起多个线程制造 ident 复用的机会：detach 之后不该有任何继承
        for i in range(5):
            t2 = threading.Thread(target=lambda i=i: print(f"未登记的线程 {i}"))
            t2.start()
            t2.join(5)

    text = log_file.read_text(encoding="utf-8")
    assert "已登记的推理线程" in text
    assert "未登记的线程" not in text


def test_concurrent_captures_do_not_cross_talk(tmp_path):
    """两个素材同时跑 S2（锁是 per-asset 的）：两份日志必须各写各的。"""
    b, c = tmp_path / "b.log", tmp_path / "c.log"
    barrier = threading.Barrier(2, timeout=10)

    def worker(path, tag):
        with lc.capture(path, header=f"==== {tag} ===="):
            for i in range(20):
                print(f"{tag} line {i}")
                if i == 9:
                    barrier.wait()      # 强制两个 capture 真正重叠一段时间

    tb = threading.Thread(target=worker, args=(b, "B"))
    tc = threading.Thread(target=worker, args=(c, "C"))
    tb.start()
    tc.start()
    tb.join(20)
    tc.join(20)

    bt, ct = b.read_text(encoding="utf-8"), c.read_text(encoding="utf-8")
    assert "B line 19" in bt and "C line" not in bt
    assert "C line 19" in ct and "B line" not in ct


def test_restores_streams_and_registry_on_normal_exit(tmp_path):
    before_out, before_err = sys.stdout, sys.stderr
    with lc.capture(tmp_path / "x.log"):
        assert sys.stdout is not before_out        # 期间确实装了 Tee
        assert sys.stderr is not before_err

    assert sys.stdout is before_out
    assert sys.stderr is before_err
    assert len(lc._writers) == 0                   # 不留僵尸登记


def test_restores_streams_when_body_raises(tmp_path):
    """推理抛错时也要还原 sys 流，且抛错前的输出已落盘。"""
    before_out = sys.stdout
    log_file = tmp_path / "y.log"
    with pytest.raises(RuntimeError), lc.capture(log_file):
        print("抛错前的一行")
        raise RuntimeError("boom")

    assert sys.stdout is before_out
    assert "抛错前的一行" in log_file.read_text(encoding="utf-8")
    assert len(lc._writers) == 0


def test_note_appends_after_capture_closed(tmp_path):
    """capture 之外还能补记一行（失败回滚时用）。"""
    log_file = tmp_path / "z.log"
    with lc.capture(log_file):
        print("正文")
    lc.note(log_file, "==== 失败：补记 ====")

    text = log_file.read_text(encoding="utf-8")
    assert text.index("正文") < text.index("==== 失败：补记 ====")


# --------------------------------------------------------------------------- #
# 心跳：实测骨架阶段（HF generate 单 batch 自回归）+26s→+226s 整整 200s 零输出，
# 光接输出仍看不出是在算还是挂了，故静默超过阈值必须自己补行。
# --------------------------------------------------------------------------- #
@pytest.fixture
def fast_tick(monkeypatch):
    """把心跳线程的巡检间隔压到 20ms，否则测试得真等一秒。"""
    monkeypatch.setattr(lc, "_HEARTBEAT_TICK", 0.02)


def test_heartbeat_fills_silent_gap_with_phase(tmp_path, fast_tick):
    """静默期要按 heartbeat_s 持续补行，并带上 set_phase 的阶段标签。"""
    log_file = tmp_path / "hb.log"
    with lc.capture(log_file, heartbeat_s=0.2):
        lc.set_phase("UNIRIG_SKELETON 骨架自回归生成")
        time.sleep(0.8)

    beats = [ln for ln in log_file.read_text(encoding="utf-8").splitlines()
             if "静默" in ln]
    assert len(beats) >= 2, "静默期只打了一次（或没打）心跳"
    assert all("UNIRIG_SKELETON" in b for b in beats)   # 看得出卡在哪一步
    assert all(b.startswith("[+") for b in beats)       # 带相对时间戳


def test_heartbeat_stays_quiet_while_output_flows(tmp_path, fast_tick):
    """有输出时不插心跳：每写一行就重置静默计时，否则真实进度会被淹掉。"""
    log_file = tmp_path / "hb2.log"
    with lc.capture(log_file, heartbeat_s=0.3):
        for _ in range(15):
            print("tick")
            time.sleep(0.05)

    text = log_file.read_text(encoding="utf-8")
    assert text.count("tick") == 15
    assert "静默" not in text


def test_heartbeat_can_be_disabled(tmp_path, fast_tick):
    log_file = tmp_path / "hb3.log"
    with lc.capture(log_file, heartbeat_s=0):
        time.sleep(0.3)
    assert "静默" not in log_file.read_text(encoding="utf-8")


def test_heartbeat_stops_after_capture_closed(tmp_path, fast_tick):
    """capture 退出后心跳线程不得再往已关闭的文件里写（否则 ValueError 刷屏）。"""
    log_file = tmp_path / "hb4.log"
    with lc.capture(log_file, heartbeat_s=0.1):
        lc.set_phase("X")
        time.sleep(0.3)
    size_at_close = log_file.stat().st_size
    time.sleep(0.3)
    assert log_file.stat().st_size == size_at_close
