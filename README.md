# pairwise-arrow-ballistics-wind

## 领域

游戏、图形与多媒体

## 主题

弓箭弹道与风偏

## 说明

逐帧风场、距离步扫掠与逐发射者预测。

## 六个缺陷与当前实现

| # | 缺陷 | 实现 | 可观测证据 |
|---|------|------|------------|
| 1 | 风场只在开火瞬间采样一次 | `wind_each_frame` / `WindField.wind_at` 每帧取新值，`_simulate_landing` 每帧重新采样 | `WindUpdate.max_delta`、`metrics()["wind"]` |
| 2 | 异步气象返回前的旧风直接当新风貌用 | `tag_sample_lag` / `AsyncWeatherService` 的读数带 `sample_frame`、`lag_frames`、`is_stale`；超阈值走 `forecast_fallback` | `WindReading.tag`、`fallback()` |
| 3 | 固定 0.1s 扫掠步长让快箭穿透薄靶 | `sweep_by_distance` 按**距离**切子步，步长还被钳到目标厚度以内，并有 `MAX_SWEEP_SUBSTEPS` 上限 | `SweepResult.step_count`、`thin_target_safe` |
| 4 | 命中后延迟回收，尾迹仍引用箭矢变换 | `Trail` 存世界坐标快照；`ArrowPool.release` 先 `detach` 再延迟一帧回收 | `trail_with_arrow()` 的 `trail_survived_recycle` |
| 5 | 同帧多人射箭共用一份预测缓存 | `PredictionCache` 以发射者为 key，`predict_per_shooter` 逐人写槽 | `PredictionCache.overwrites == 0` |
| 6 | 弹道随帧长变化而改变 | `integrate_fixed_step` 固定 `dt` + 累加器，余量跨帧携带 | `determinism().varying_frame_dt_same_result` |

`validate_ballistic()`、`audit()`、`metrics()` 把上面六项串成一次性自检；`determinism()`
校验同 seed 复现与帧率无关。

运行测试：

```bash
python -m pytest -q
```

- `tests/test_arrow_ballistics_wind.py`：契约测试（15 项，所有入口非 `None`）。
- `tests/test_ballistics_behaviors.py`：六个缺陷的行为回归测试。
