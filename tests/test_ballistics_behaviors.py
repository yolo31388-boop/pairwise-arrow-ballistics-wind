"""六个缺陷的行为回归测试（契约测试之外的补充）。"""

import math

import pytest

import gameplay.arrow_ballistics_wind as m


def test_wind_changes_every_frame():
    update = m.wind_each_frame(frames=20, seed=3)
    assert update.per_frame and not update.frozen_wind
    winds = [s.wind for s in update.samples]
    assert len(set(winds)) == len(winds)
    assert update.max_delta > 1e-3


def test_freeze_wind_versus_per_frame_differs():
    """整段飞行只采样开火瞬间的风，会比逐帧风产生可观测的落点差。"""
    field = m.WindField(seed=5, gust=8.0)
    origin = (0.0, 1.5, 0.0)
    velocity = m._vacuum_solution(origin, m.Target(center=(0.0, 1.5, 40.0)), 25.0)
    per_frame, _, _ = m._simulate_landing(origin, velocity, field, stop_z=40.0)
    frozen_wind = field.wind_at(0.0)

    class _Frozen:
        def wind_at(self, _t):
            return frozen_wind

    frozen, _, _ = m._simulate_landing(origin, velocity, _Frozen(), stop_z=40.0)
    assert m._vdist(per_frame, frozen) > 0.05


@pytest.mark.parametrize("lag,stale", [(0, False), (1, False), (3, True), (8, True)])
def test_sample_lag_is_tagged(lag, stale):
    reading = m.tag_sample_lag(lag_frames=lag)
    assert reading.lag_frames == lag
    assert reading.lag_seconds == pytest.approx(lag * m.DEFAULT_FRAME_DT)
    assert reading.is_stale is stale
    assert reading.fallback_used is stale
    assert str(lag) in reading.tag


def test_stale_readings_fall_back_to_current_forecast():
    result = m.fallback(lag_frames=5)
    assert result.stale and result.fallback_used and result.used_forecast
    assert result.wind == result.forecast_wind
    assert not result.used_previous_frame


def test_distance_sweep_hits_thin_target_a_time_step_would_tunnel():
    # 55 m/s × 0.1 s ≈ 5.5 m，远大于 3 cm 的薄靶
    speed, frame_dt, thickness = 55.0, 0.1, 0.03
    step_distance = speed * frame_dt
    assert step_distance > thickness
    start = (0.0, 1.5, 20.0 - step_distance * 0.5)
    end = (0.0, 1.5, 20.0 + step_distance * 0.5)

    result = m.sweep_by_distance(
        step=0.1, start=start, end=end, target_center=(0.0, 1.5, 20.0), thickness=thickness
    )
    assert result.distance_based and not result.time_based
    assert result.hit
    assert result.point[2] == pytest.approx(20.0, abs=thickness)
    assert result.step_count > 1


def test_sweep_caps_absurd_substep_counts():
    # 亚毫米步长 × 2 km 线段本会产生 2000 万子步；必须被钳制并仍然完成。
    r = m.sweep_by_distance(step=1e-9, start=(0.0, 50.0, -1000.0),
                            end=(0.0, 50.0, 1000.0),
                            target_center=(0.0, 1.5, 20.0))
    assert r.capped and r.step_count == m.MAX_SWEEP_SUBSTEPS
    assert not r.hit  # 目标不在线段上


def test_trail_snapshots_survive_arrow_recycle():
    result = m.trail_with_arrow(frame=10)
    assert result.trail_frozen
    assert result.trail_survived_recycle
    assert not result.trail_origin_glitch
    assert result.recycled_after_one_frame
    assert not result.recycled_on_hit_frame
    # 箭矢确实被池重置到原点，但尾迹点仍留在世界空间
    assert result.arrow_position_after_recycle == (0.0, 0.0, 0.0)
    assert result.trail_min_distance_from_origin > 1.0


def test_pool_recycles_one_frame_later():
    pool = m.ArrowPool()
    arrow = pool.acquire("p1", (0.0, 1.5, 0.0), (0.0, 0.0, 55.0))
    assert pool.release(arrow, 4) == 5
    pool.step(4)
    assert not arrow.recycled and pool.pending == 1
    pool.step(5)
    assert arrow.recycled and pool.free == 1 and pool.pending == 0


def test_prediction_cache_is_keyed_by_shooter():
    cache = m.predict_per_shooter(shooters=4, frame=12, seed=2)
    assert len(cache) == 4
    assert cache.overwrites == 0
    assert cache.keyed_by_shooter()
    assert set(cache._slots) == {"p1", "p2", "p3", "p4"}
    # 同一帧，各发射者的预测圈互不覆盖
    assert all(p.frame == 12 for p in cache._slots.values())
    assert len({(p.landing[0], p.landing[1]) for p in cache._slots.values()}) == 4


def test_fixed_step_is_frame_rate_independent():
    a = m._integrate_frames((1 / 60, 1 / 60))
    b = m._integrate_frames((1 / 30,))
    c = m._integrate_frames((1 / 120,) * 4)
    assert a["steps"] == b["steps"] == c["steps"]
    assert m._vdist(a["end"], b["end"]) < 1e-9
    assert m._vdist(a["end"], c["end"]) < 1e-9


def test_leftover_is_carried_not_dropped():
    r = m.integrate_fixed_step(duration=1 / 60, fixed_dt=1 / 240)
    assert r.steps == 4
    assert abs(r.leftover) < 1e-12
    partial = m.integrate_fixed_step(duration=0.4 * (1 / 240), fixed_dt=1 / 240)
    assert partial.steps == 0
    assert partial.leftover == pytest.approx(0.4 / 240)


def test_determinism_and_audit_are_green():
    d = m.determinism(seed=7)
    assert d.identical and d.fixed_step_exact and d.varying_frame_dt_same_result
    v = m.validate_ballistic(seed=7)
    assert v.passed, v.failures
    a = m.audit(seed=7)
    assert a.ok, a.issues


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), None, [], 0, -5, 10**9])
def test_entry_points_tolerate_hostile_input(value):
    for name in ("wind_each_frame", "tag_sample_lag", "sweep_by_distance",
                 "predict_per_shooter", "integrate_fixed_step", "fallback"):
        assert getattr(m, name)(value) is not None
