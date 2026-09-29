/**
 * `/v2` 接口契约的 TypeScript 镜像。
 *
 * 与后端 `src/ai3d/retarget/schemas_v2.py`、`axis_norm.build_align()` 严格对应；
 * 字段增删必须两侧同步改。面编号、语义轴这类**固定约定**优先走
 * `GET /v2/conventions` 动态取（见 `Conventions`），此处只保留结构与兜底常量。
 */

/** 素材类型：模型走四阶段，动画只走 S1 后进入素材库供复用。 */
export type AssetKind = 'model' | 'animation'

/**
 * 素材生命周期状态。`ALIGN_READY` 是 S1 完成态（等人工确认），也是 S2 的准入门槛：
 * `skin.weld_epsilon` / `foot_contact_height` 都是米制绝对阈值，单位未矫正就绑定
 * 会得到完全错误的焊接半径。
 */
export type AssetState = 'CREATED' | 'NORMALIZING' | 'ALIGN_READY' | 'READY' | 'FAILED'

/** S2 绑定状态（缓存于 bindings 表，供模型素材跨作业复用）。 */
export type BindingState = 'NONE' | 'PENDING' | 'RUNNING' | 'WAIT_VIEWS' | 'READY' | 'FAILED'

/** 四阶段分组。 */
export type StageGroup = 'S1' | 'S2' | 'S3' | 'S4'

/** 面序号 1..6（规范系外切盒：`1=+X left, 2=-X right, 3=+Y up, 4=-Y down, 5=+Z front, 6=-Z back`）。 */
export type FaceId = 1 | 2 | 3 | 4 | 5 | 6

export type Vec3 = [number, number, number]
export type Mat3 = [Vec3, Vec3, Vec3]

/**
 * 规范系（米制）轴对齐外切盒：中心 + 三轴全长。三轴恒为单位阵（AABB），
 * 且已在场景坐标下（`build_align` 先旋到规范朝向再乘 scale），前端直接画，
 * 不再需要 canon 的 rotation/scale 二次变换。
 */
export interface BboxDict {
  center: Vec3
  extents: Vec3
}

/** PCA 三主轴（模型坐标，特征值降序）；仅作调试展示与缺先验时的轴线证据。 */
export interface PcaInfo {
  eigenvalues: number[]
  axes: Vec3[]
}

/** 一个候选单位：换算系数、换算后身高、与成人中值 1.75m 的对数距离（越小越可信）。 */
export interface UnitCandidate {
  unit: string
  scale: number
  height_m: number
  log_dist: number
}

/** 单位自动推断结果。`hint` 非空表示身高落在成人区间外，需人工确认。 */
export interface AlignUnit {
  detected: string
  scale: number
  height_m: number
  candidates: UnitCandidate[]
  hint: string
}

/**
 * 自动校准结果：`up` / `forward` 为吸附到模型坐标轴的单位向量，`method` 标注
 * 先验来源（skeleton / mesh / identity），`notes` 是探测过程的诊断说明；
 * `axis_faces` 是自动结果换算成的面号指派（下拉框的默认值）。
 */
export interface AlignAuto {
  method: string
  notes: string[]
  up: Vec3
  forward: Vec3
  axis_faces: AxisFaces
}

/**
 * 面号指派：正面/左边/上面 各是外切盒几号面。编号 1..6 按**模型原坐标轴**固定
 * （1=+X 2=-X 3=+Y 4=-Y 5=+Z 6=-Z），贴在模型自身的面上、不随校准旋转改变。
 */
export interface AxisFaces {
  front: number
  left: number
  up: number
}

/**
 * 人工修正段。V2 下人工按面号指派语义轴（`axis_faces`，null = 沿用自动结果）+
 * 覆盖单位/身高；两项覆盖的**清除**用哨兵值（`null` 已被占用为「不改」）：
 * `height_override <= 0` 取消身高直填，`unit_override = ""` 取消强制单位。
 */
export interface AlignManual {
  /** 人工面号指派；null 表示未改过、沿用自动探测（等价于朝向正确）。 */
  axis_faces: AxisFaces | null
  unit_override: string | null
  height_override: number | null
}

/** 最终写入 canon 节点的旋转（模型坐标 → 规范坐标）与统一缩放。 */
export interface AlignFinal {
  rotation: Mat3
  scale: number
  changed: boolean
}

/** `align.json` 全文（S1 唯一产物）。 */
export interface AlignDoc {
  version: number
  pca: PcaInfo | null
  bbox: BboxDict | null
  unit: AlignUnit
  auto: AlignAuto
  manual: AlignManual
  /** 每个编号面（1..6）在当前规范系下的外法向：前端把编号画到对应的盒面上。 */
  face_axes: Vec3[]
  final: AlignFinal
}

/** `GET /v2/conventions`：前后端共享的固定约定，单点下发避免各写一份走偏。 */
export interface Conventions {
  align_version: number
  face_labels: string[]
  canonical_axes: { up: string; front: string; left: string }
  asset_states: AssetState[]
  asset_kinds: AssetKind[]
}

export interface BindingInfo {
  state: BindingState
  /** 子步骤级进度（RENDER_VIEWS → POSE_INFER → SOLVE_RIG → BUILD_RIG）；终态为 null。 */
  stage: string | null
  message: string | null
  /** 只在 `FAILED` 时非空，与 `AssetInfo.error`（S1 的）互不覆盖。 */
  error: string | null
  confidence: number | null
  revision: number
  has_rig: boolean
  has_skin: boolean
  /** 已回传的视角图文件名（`views/` 目录清单），前端据此判断哪些视角待补。 */
  views: string[]
  report: Record<string, unknown> | null
}

/** 列表项：不含 align/meta 大字段。 */
export interface AssetSummary {
  asset_id: string
  kind: AssetKind
  name: string | null
  filename: string | null
  state: AssetState
  binding_state: BindingState
  height_m: number | null
  unit: string | null
  error: string | null
  created_at: string | null
  updated_at: string | null
}

/** `GET /v2/assets/{id}` 响应。 */
export interface AssetInfo {
  asset_id: string
  kind: AssetKind
  name: string | null
  filename: string | null
  state: AssetState
  align: AlignDoc | null
  meta: Record<string, unknown> | null
  binding: BindingInfo
  error: string | null
  created_at: string | null
  updated_at: string | null
}

export interface AssetListResponse {
  assets: AssetSummary[]
  total: number
  limit: number
  offset: number
}

export interface CreateAssetResponse {
  asset_id: string
  state: AssetState
  message: string
}

/** S1 产物回执（上传后自动跑、PATCH 重算共用）。 */
export interface AlignResponse {
  asset_id: string
  state: AssetState
  filename: string | null
  align: AlignDoc | Record<string, never>
  error: string | null
  message: string
}

/**
 * `PATCH /v2/assets/{id}/align` 请求体。留 `undefined` 的字段表示不改该项；
 * `reset=true` 清空全部人工修正回到自动探测。人工能改的只有 `axis_faces` 面号指派
 * 与单位/身高覆盖，校准基与外切盒每次由后端重测。
 */
export interface AlignPatch {
  axis_faces?: AxisFaces | null
  unit_override?: string | null
  height_override?: number | null
  reset?: boolean
}

export interface HealthResponse {
  status: string
  version: string
  assimp: string | null
  counts: Record<string, number>
}

export interface DeleteAssetResponse {
  status: string
  asset_id: string
  references: { as_model: number; as_animation: number }
}

// --------------------------------------------------------------------------- //
// S2 绑定（素材级产物，跨作业复用）
// --------------------------------------------------------------------------- //
/** 一个离屏渲染相机：`view_id` 同时是回传图片的文件名主干（后端按 `p.stem` 认）。 */
export interface CameraSpec {
  view_id: string
  name: string
  /** 绕 up 轴方位角（度）。 */
  azimuth: number
  /** 仰角（度）。 */
  elevation: number
}

/**
 * `GET /v2/assets/{id}/view_spec` 下发的渲染规格。
 *
 * 前端**必须**按这份规格渲染（方位角/仰角/画布尺寸/正交与否）：后端三角化会用
 * 同一份 `view_spec.json` 重建相机，两边各写一套默认值会让重建关节整体错位。
 */
export interface ViewSpec {
  model_url: string
  cameras: CameraSpec[]
  width: number
  height: number
  /** Three.js/GLB 世界 up（固定 `"Y"`）。 */
  up_axis: string
  background: string
  ortho: boolean
}

/** 关节微调项：留空的字段表示不改该项。 */
export interface JointPatch {
  head?: number[] | null
  tail?: number[] | null
  locked?: boolean | null
  confidence?: number | null
}

/** `PATCH /v2/assets/{id}/rig` 请求体；`revision` 是乐观锁，对不上后端回 409。 */
export interface RigPatch {
  revision: number
  joints: Record<string, JointPatch>
}

/** `rig.json` 里的单个关节（`Rig.to_dict()`）。 */
export interface RigJoint {
  semantic_id: string
  /** 父关节的 semantic_id；pelvis 为 null。 */
  parent: string | null
  head: number[]
  confidence: number
  /** `proportion` 比例先验 / `solved` 多视角三角化 / `manual` 人工改过。 */
  source: string
  locked: boolean
}

export interface RigDoc {
  revision: number
  height: number
  joints: Record<string, RigJoint>
}

/** `skin_report.json`（`skinning._skin_report` 的实测键，无 `vertices`）。 */
export interface SkinReport {
  zero_rows: number
  island_stats: { components: number; largest_share: number; tiny_lt16: number }
  weld: { clusters: number; merged_verts: number }
  components: { count: number; fallback: number; proxy: number; transferred: number }
  weight_jump: { mean: number; max: number }
  strain: { med: number; max: number; gt20pct: number; edges: number }
  seconds: number
  [key: string]: unknown
}

export interface ViewListResponse {
  asset_id: string
  views: string[]
  total: number
  /** 与 `views` 一一对应的下载路径（带 `/v2` 前缀）。 */
  urls: string[]
}

/** `POST /bind` 与 `POST /rebind` 的回执：S2 已排到后台线程。 */
export interface BindResponse {
  status: string
  asset_id: string
  pose_ai: boolean
  binding: BindingInfo
  message: string
}

/** `POST /views` 的回执；`resumed=true` 表示它把挂在 WAIT_VIEWS 的 S2 续跑了。 */
export interface UploadViewsResponse {
  status: string
  asset_id: string
  views: string[]
  total: number
  resumed: boolean
  message: string
  binding: BindingInfo
}

/** `PATCH /rig` 的回执；`reskin="skipped"` 表示重算蒙皮没起来（多半是已在跑）。 */
export interface PatchRigResponse {
  status: string
  asset_id: string
  rig_revision: number
  reskin: 'started' | 'skipped' | string
  binding: BindingInfo
}

/** `POST /reskin` 的回执：只重算蒙皮，骨架 revision 不变。 */
export interface ReskinResponse {
  status: string
  asset_id: string
  reskin: boolean
  binding: BindingInfo
  message: string
}

// --------------------------------------------------------------------------- //
// 作业（S2~S4）—— 路由在 P3~P5 接入，契约已定（后端 schemas_v2.JobInfoV2）
// --------------------------------------------------------------------------- //
/** 旧九阶段作为 S1~S4 的内部子步骤保留。 */
export type Stage =
  | 'NORMALIZE'
  | 'PRECHECK'
  | 'RENDER_VIEWS'
  | 'POSE_INFER'
  | 'SOLVE_RIG'
  | 'BUILD_RIG'
  | 'MAP_SOURCE'
  | 'RETARGET'
  | 'EXPORT_VERIFY'

export type StageStatus = 'PENDING' | 'RUNNING' | 'WAITING' | 'DONE' | 'FAILED' | 'SKIPPED'

export interface StageInfo {
  stage: Stage
  status: StageStatus
  message: string | null
  started_at: string | null
  finished_at: string | null
}

export interface ArtifactInfo {
  kind: string
  path: string
  created_at: string | null
}

/** `GET /v2/jobs/{id}` 响应。 */
export interface JobInfoV2 {
  job_id: string
  name: string | null
  state: string
  stage_group: StageGroup | null
  current_stage: Stage | null
  model_asset_id: string
  anim_asset_id: string | null
  mapping_revision: number
  review_reasons: string[]
  error: string | null
  created_at: string | null
  updated_at: string | null
  stages: StageInfo[]
  artifacts: ArtifactInfo[]
}
