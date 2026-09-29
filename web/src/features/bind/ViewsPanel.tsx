/**
 * 多视角面板：展示已回传的视角图，并触发「离屏渲染 → 回传」。
 *
 * 视角图是 S2 的 AI 路径唯一输入：后端 `RENDER_VIEWS` 阶段只写规格不渲染（渲染要
 * GPU，服务端没有），拿到图才继续 `POSE_INFER`。所以 `WAIT_VIEWS` 不是失败，而是
 * 在等这个按钮。
 *
 * 缩略图按 `view_id` 与相机规格配对：文件名主干就是 view_id（后端 `p.stem` 认），
 * 配不上的图会在表格里显示为「多余」—— 那种图不参与三角化，多半是上一次残留。
 */
import type { CameraSpec } from '../../api/types'
import { viewImageUrl } from '../../api/client'

export interface ViewsPanelProps {
  assetId: string
  /** `binding.views`：后端 `views/` 目录里的文件名清单。 */
  views: string[]
  /** `view_spec.cameras`；还没拉到就只列文件名。 */
  cameras: CameraSpec[] | null
  /** 缩略图的缓存 bust 版本键（用素材的 `updated_at`）。 */
  version?: string | null
  busy?: boolean
  disabled?: boolean
  progress: string | null
  onCapture: () => void
}

function stem(name: string): string {
  const i = name.lastIndexOf('.')
  return i > 0 ? name.slice(0, i) : name
}

export function ViewsPanel({
  assetId,
  views,
  cameras,
  version = null,
  busy = false,
  disabled = false,
  progress,
  onCapture,
}: ViewsPanelProps) {
  const byStem = new Map(views.map((name) => [stem(name), name]))
  const matched = cameras ? cameras.map((c) => byStem.get(c.view_id) ?? null) : []
  const known = new Set(cameras?.map((c) => c.view_id) ?? [])
  const extra = views.filter((name) => !known.has(stem(name)))

  return (
    <div className="card">
      <h3>多视角（{views.length}{cameras ? ` / ${cameras.length}` : ''}）</h3>
      <div className="muted" style={{ marginBottom: 6 }}>
        浏览器按后端下发的正交相机规格离屏渲染归一化模型并回传；后端用 DWPose 逐张
        提 2D 关键点，再按同一套相机参数三角化出三维关节。
      </div>

      {cameras ? (
        <div className="row tight" style={{ flexWrap: 'wrap', gap: 6 }}>
          {cameras.map((cam, i) => {
            const name = matched[i]
            return (
              <div key={cam.view_id} style={{ textAlign: 'center' }}>
                <div
                  style={{
                    width: 58, height: 58, borderRadius: 6, overflow: 'hidden',
                    border: `1px solid ${name ? 'var(--line)' : 'var(--err)'}`,
                    background: 'var(--input)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}
                >
                  {name ? (
                    <img
                      src={viewImageUrl(assetId, name, version)}
                      alt={cam.view_id}
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    />
                  ) : (
                    <span className="muted">缺</span>
                  )}
                </div>
                <div className="muted" style={{ marginTop: 2 }}>{cam.view_id}</div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="muted">
          {views.length ? views.join('、') : '还没有视角图。'}
        </div>
      )}

      {extra.length ? (
        <div className="note warn">
          有 {extra.length} 张图的文件名对不上任何相机（{extra.join('、')}），
          它们不会参与三角化。重新渲染一次即可覆盖。
        </div>
      ) : null}

      <div className="row" style={{ marginTop: 8 }}>
        <button disabled={disabled || busy} onClick={onCapture}>
          {busy ? (progress ?? '渲染中…') : '📸 渲染并回传多视角'}
        </button>
      </div>
      {progress && !busy ? <div className="muted">{progress}</div> : null}
    </div>
  )
}

export default ViewsPanel
