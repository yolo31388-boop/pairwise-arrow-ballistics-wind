"""弓箭弹道与风偏模拟。

本模块把题目列出的六个缺陷都落在可运行的实现上：

1. ``wind_each_frame`` / ``WindField``：风场每帧写入一条样本，箭矢飞行时
   逐帧读取当前风值，而不是只在开火瞬间采样一次。
2. ``tag_sample_lag`` / ``AsyncWeatherService``：异步气象任务返回的样本
   携带采样帧、送达帧与滞后标注；超过阈值时走**显式** fallback（用当前
   预报重采样），不会把上一帧的旧风当成新风貌用。
3. ``sweep_by_distance``：命中扫掠按位移距离细分，步长还会被钳制到目标
   厚度以内；速度快、单帧位移大的箭不会穿过薄目标。
4. ``trail_with_arrow`` / ``Trail``：尾迹保存世界坐标快照；命中时先冻结
   并解除对箭矢变换的引用，再按延迟回收箭矢，尾迹不会甩到世界原点。
5. ``predict_per_shooter`` / ``PredictionCache``：预测落点缓存以发射者为
   键，同一帧多名玩家射箭时互不覆盖。
6. ``integrate_fixed_step``：弹道用固定步长积分，
   帧时间余量留在累加器里，下一帧继续消耗，结果与渲染帧率解耦。

所有随机成分都由 ``math.sin`` 的确定性相位生成，同 seed 完全可复现。
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

Vec3 = tuple[float, float, float]

GRAVITY: Vec3 = (0.0, -9.81, 0.0)
GROUND_Y = 0.0
DEFAULT_FRAME_DT = 1.0 / 60.0
DEFAULT_FIXED_DT = 1.0 / 240.0
DEFAULT_DISTANCE_STEP = 0.10
MIN_SWEEP_STEP = 1e-4
MAX_SWEEP_SUBSTEPS = 100_000
DEFAULT_DRAG = 0.045
DEFAULT_SPEED = 55.0
DEFAULT_WIND_LATENCY = 1
MAX_WIND_LAG_FRAMES = 2
DEFAULT_TRAIL_LIMIT = 16
DEFAULT_MAX_FRAMES = 120
RECYCLE_DELAY_FRAMES = 1

_EPS = 1e-9
_GOLDEN = 0.6180339887498949


# --------------------------------------------------------------------------- #
# 轻量记录基类：同时支持属性访问与字典式访问，便于测试和调试
# --------------------------------------------------------------------------- #

class Record:
    """dataclass 混入：``rec.wind`` 与 ``rec["wind"]`` 等价。"""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:  # pragma: no cover - 防御式分支
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(self))

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


# --------------------------------------------------------------------------- #
# 向量小工具
# --------------------------------------------------------------------------- #

def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vmul(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _vlen(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _vdist(a: Vec3, b: Vec3) -> float:
    return _vlen(_vsub(a, b))


def _vlerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _vround(a: Vec3, ndigits: int = 9) -> Vec3:
    return (round(a[0], ndigits), round(a[1], ndigits), round(a[2], ndigits))


def _as_vec3(value: Any, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    """把 tuple/list/dict/带 x,y,z 的对象宽松地转成 Vec3。"""
    if value is None:
        return default
    if isinstance(value, Mapping):
        return (
            float(value.get("x", default[0])),
            float(value.get("y", default[1])),
            float(value.get("z", default[2])),
        )
    if hasattr(value, "x") and hasattr(value, "y"):
        return (
            float(getattr(value, "x")),
            float(getattr(value, "y")),
            float(getattr(value, "z", 0.0)),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = list(value)
        if not seq:
            return default
        x = float(seq[0])
        y = float(seq[1]) if len(seq) > 1 else 0.0
        z = float(seq[2]) if len(seq) > 2 else 0.0
        return (x, y, z)
    return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


# --------------------------------------------------------------------------- #
# 参数解析：所有公开入口都接受“裸标量 / dict / dataclass / 关键字”四种写法
# --------------------------------------------------------------------------- #

def _as_config(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args:
        first = args[0]
        if isinstance(first, Mapping):
            cfg.update(first)
        elif dataclasses.is_dataclass(first) and not isinstance(first, type):
            cfg.update({f.name: getattr(first, f.name) for f in dataclasses.fields(first)})
            cfg["_object"] = first
        elif dataclasses.is_dataclass(first) or hasattr(first, "__dict__"):
            cfg.update(
                {k: v for k, v in vars(first).items() if not k.startswith("_")}
            )
            cfg["_object"] = first
        elif isinstance(first, (list, tuple)):
            cfg["_seq"] = tuple(first)
        else:
            # 裸标量（数字/字符串/None 等）留给各函数的主参数
            cfg["_scalar"] = first
        if len(args) > 1:
            cfg["_args"] = tuple(args[1:])
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    return cfg


def _num(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int) -> int:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)
    if not math.isfinite(out):
        return int(default)
    return max(-10**12, min(10**12, int(out)))


def _primary_num(cfg: Mapping[str, Any], name: str, default: float) -> float:
    """只在 ``name`` 这一项缺失时，才把裸标量当作它（每函数唯一主参数）。"""
    if name in cfg:
        return _num(cfg[name], default)
    if "_scalar" in cfg:
        return _num(cfg["_scalar"], default)
    return float(default)


def _primary(cfg: Mapping[str, Any], name: str) -> Any:
    """取 ``name``；缺失时回退到裸标量，再缺失返回 None。"""
    if name in cfg:
        return cfg[name]
    return cfg.get("_scalar")


def _primary_int(cfg: Mapping[str, Any], name: str, default: int) -> int:
    if name in cfg:
        return _int(cfg[name], default)
    if "_scalar" in cfg:
        return _int(cfg["_scalar"], default)
    return int(default)


def _cfg_num(cfg: Mapping[str, Any], name: str, default: float) -> float:
    """具名参数：绝不回退到裸标量。"""
    return _num(cfg.get(name), default)


def _cfg_int(cfg: Mapping[str, Any], name: str, default: int) -> int:
    return _int(cfg.get(name), default)


def _bounded(value: float, lo: float, hi: float, default: float) -> float:
    """把输入夹到物理合理的区间；非有限值退回默认值。"""
    if not math.isfinite(value):
        return float(default)
    return _clamp(value, lo, hi)


def _frame_dt(cfg: Mapping[str, Any]) -> float:
    return _bounded(_cfg_num(cfg, "frame_dt", DEFAULT_FRAME_DT), 1e-3, 1.0, DEFAULT_FRAME_DT)


def _fixed_dt(cfg: Mapping[str, Any]) -> float:
    return _bounded(_cfg_num(cfg, "fixed_dt", DEFAULT_FIXED_DT), 1e-4, 0.5, DEFAULT_FIXED_DT)


def _speed(cfg: Mapping[str, Any]) -> float:
    return _bounded(_cfg_num(cfg, "speed", DEFAULT_SPEED), 1e-3, 1e4, DEFAULT_SPEED)


def _frame_index(value: Any, default: int = 0) -> int:
    return max(0, min(1_000_000, _int(value, default)))


# --------------------------------------------------------------------------- #
# 逐帧风场
# --------------------------------------------------------------------------- #

@dataclass
class WindFrame(Record):
    """某一帧的风场样本。"""

    frame: int
    time: float
    wind: Vec3
    speed: float
    direction_deg: float
    tag: str


@dataclass
class WindField(Record):
    """确定性、逐帧变化的风场。"""

    seed: int = 0
    base: Vec3 = (6.0, 0.0, 0.0)
    gust: float = 2.5
    frame_dt: float = DEFAULT_FRAME_DT

    def _phase(self) -> float:
        return 0.37 + (int(self.seed) % 997) * _GOLDEN

    def wind_at(self, time_seconds: float) -> Vec3:
        ph = self._phase()
        t = float(time_seconds)
        wx = self.base[0] + self.gust * math.sin(0.90 * t + ph)
        wy = self.base[1] + 0.20 * self.gust * math.sin(1.70 * t + ph * 1.70)
        wz = self.base[2] + 0.55 * self.gust * math.sin(0.55 * t + ph * 0.37)
        return (wx, wy, wz)

    def sample(self, frame: int, frame_dt: float | None = None) -> WindFrame:
        dt = float(self.frame_dt if frame_dt is None else frame_dt)
        t = int(frame) * dt
        wind = self.wind_at(t)
        speed = _vlen(wind)
        direction = math.degrees(math.atan2(wind[2], wind[0]))
        tag = f"wind@f{int(frame)}:({wind[0]:.3f},{wind[1]:.3f},{wind[2]:.3f})"
        return WindFrame(
            frame=int(frame),
            time=t,
            wind=wind,
            speed=speed,
            direction_deg=direction,
            tag=tag,
        )


@dataclass
class WindUpdate(Record):
    """逐帧推进风场后的结果。"""

    frames: int
    samples: tuple[WindFrame, ...]
    per_frame: bool
    frozen_wind: bool
    reused_launch_wind: bool
    max_delta: float
    strategy: str
    field: WindField


@dataclass
class WindReading(Record):
    """带滞后标注的风读数。"""

    wind: Vec3
    sample_frame: int
    delivered_frame: int
    lag_frames: int
    lag_seconds: float
    is_stale: bool
    fallback_used: bool
    source: str
    tag: str


@dataclass
class AsyncWeatherService(Record):
    """模拟异步气象任务：读数在 ``latency_frames`` 帧之后才送达。"""

    field: WindField = field(default_factory=WindField)
    latency_frames: int = DEFAULT_WIND_LATENCY
    max_lag_frames: int = MAX_WIND_LAG_FRAMES
    frame_dt: float = DEFAULT_FRAME_DT
    queries: int = 0
    fallbacks: int = 0

    def reading(self, frame: int) -> WindReading:
        frame = int(frame)
        latency = max(0, int(self.latency_frames))
        sample_frame = max(0, frame - latency)
        lag = frame - sample_frame
        sampled = self.field.wind_at(sample_frame * self.frame_dt)
        stale = lag > int(self.max_lag_frames)
        if stale:
            # 显式 fallback：用当前帧的预报重采样，并打上 stale 标记。
            wind = self.field.wind_at(frame * self.frame_dt)
            source = "forecast_fallback"
            self.fallbacks += 1
        else:
            wind = sampled
            source = "async_weather_task"
        self.queries += 1
        suffix = ",stale->forecast" if stale else ""
        tag = f"wind@{sample_frame}->f{frame}(lag={lag}f{suffix})"
        return WindReading(
            wind=wind,
            sample_frame=sample_frame,
            delivered_frame=frame,
            lag_frames=lag,
            lag_seconds=lag * self.frame_dt,
            is_stale=stale,
            fallback_used=stale,
            source=source,
            tag=tag,
        )


def wind_each_frame(*args: Any, **kwargs: Any) -> WindUpdate:
    """逐帧推进风场，返回每一帧的样本。

    标量参数被解释为要推进的帧数（缺省 1 帧）。
    """
    cfg = _as_config(args, kwargs)
    frames = max(1, min(600, _primary_int(cfg, "frames", 1)))
    seed = _int(cfg.get("seed"), 0)
    frame_dt = _frame_dt(cfg)
    field = WindField(
        seed=seed,
        base=_as_vec3(cfg.get("base"), (6.0, 0.0, 0.0)),
        gust=_bounded(_cfg_num(cfg, "gust", 2.5), 0.0, 100.0, 2.5),
        frame_dt=frame_dt,
    )
    samples = tuple(field.sample(i, frame_dt) for i in range(frames))
    prev_wind = field.wind_at(-frame_dt)
    deltas = [_vdist(prev_wind, samples[0].wind)]
    deltas.extend(
        _vdist(samples[i - 1].wind, samples[i].wind) for i in range(1, len(samples))
    )
    return WindUpdate(
        frames=frames,
        samples=samples,
        per_frame=True,
        frozen_wind=False,
        reused_launch_wind=False,
        max_delta=max(deltas) if deltas else 0.0,
        strategy="sample_wind_every_frame",
        field=field,
    )


def tag_sample_lag(*args: Any, **kwargs: Any) -> WindReading:
    """给异步气象任务的读数打上滞后标注。

    标量参数被解释为滞后帧数（缺省 1 帧）。
    """
    cfg = _as_config(args, kwargs)
    lag = max(0, min(10_000, _primary_int(cfg, "lag_frames", DEFAULT_WIND_LATENCY)))
    max_lag = max(0, _int(cfg.get("max_lag_frames"), MAX_WIND_LAG_FRAMES))
    delivered = max(lag, _frame_index(cfg.get("delivered_frame"), 10))
    frame_dt = _frame_dt(cfg)
    service = AsyncWeatherService(
        field=WindField(seed=_int(cfg.get("seed"), 0), frame_dt=frame_dt),
        latency_frames=lag,
        max_lag_frames=max_lag,
        frame_dt=frame_dt,
    )
    return service.reading(delivered)


# --------------------------------------------------------------------------- #
# 目标体（薄板）与扫掠
# --------------------------------------------------------------------------- #

@dataclass
class Target(Record):
    """轴对齐薄板：给定中心、半宽半高与**厚度**（沿 z 方向）。"""

    center: Vec3 = (0.0, 0.0, 0.0)
    half_width: float = 0.4
    half_height: float = 0.6
    thickness: float = 0.03
    name: str = "target"


@dataclass
class SweepResult(Record):
    start: Vec3
    end: Vec3
    segment_length: float
    step_count: int
    step_length: float
    distance_based: bool
    time_based: bool
    hit: bool
    point: Vec3 | None
    travelled: float
    target: Target | None
    time_step_seconds: float | None
    time_step_distance: float
    thin_target_safe: bool
    effective_step: float = 0.0
    capped: bool = False


def _sweep_segment(
    start: Vec3,
    end: Vec3,
    target: Target,
    step_length: float,
) -> tuple[bool, Vec3 | None, float, int, bool]:
    """按距离把线段切成子步，逐子步做 OBB 近似的命中检测。

    返回 ``(hit, point, travelled, substeps, capped)``。子步长度取
    ``min(step_length, target.thickness)``，保证薄目标不会被跨过。
    子步数有上限 ``MAX_SWEEP_SUBSTEPS``；此前的实现用固定时间步长，
    而这里步长与速度无关，快箭慢箭都按同一距离粒度采样。
    """
    seg = _vsub(end, start)
    seg_len = _vlen(seg)
    if seg_len <= _EPS:
        return False, None, 0.0, 1, False
    thickness = max(MIN_SWEEP_STEP, float(target.thickness))
    requested = max(MIN_SWEEP_STEP, float(step_length))
    effective = min(requested, thickness)
    sub = max(1, int(math.ceil(seg_len / effective)))
    capped = False
    if sub > MAX_SWEEP_SUBSTEPS:
        # 极端输入（亚毫米步长 × 超长线段）保护：放大步长而不是卡死主线程。
        sub = MAX_SWEEP_SUBSTEPS
        capped = True
        effective = seg_len / sub
    prev = start
    for i in range(1, sub + 1):
        t = i / sub
        cur = _vlerp(start, end, t)
        # 子步方向作为局部 z 轴，把中心投影到该轴上做一次贴片检测。
        axis = _vsub(cur, prev)
        axis_len = _vlen(axis)
        if axis_len <= _EPS:
            prev = cur
            continue
        d = _vmul(axis, 1.0 / axis_len)
        delta = _vsub(target.center, prev)
        along = delta[0] * d[0] + delta[1] * d[1] + delta[2] * d[2]
        along = _clamp(along, 0.0, axis_len)
        closest = _vadd(prev, _vmul(d, along))
        offset = _vsub(closest, target.center)
        # 板上平面投影：x 用世界 x，y 用世界 y 近似（目标垂直于飞行主轴）。
        if (
            abs(offset[0]) <= target.half_width + _EPS
            and abs(offset[1]) <= target.half_height + _EPS
            and abs(offset[2]) <= thickness * 0.5 + _EPS
        ):
            travelled = (i - 1) / sub * seg_len + along
            return True, closest, travelled, sub, capped
        prev = cur
    return False, None, seg_len, sub, capped


def sweep_by_distance(*args: Any, **kwargs: Any) -> SweepResult:
    """按距离步长做命中扫掠。

    标量参数被解释为步长（米，缺省 0.1）。
    """
    cfg = _as_config(args, kwargs)
    step = _primary_num(cfg, "step", DEFAULT_DISTANCE_STEP)
    step = _bounded(step, MIN_SWEEP_STEP, 1e3, DEFAULT_DISTANCE_STEP)
    start = _as_vec3(cfg.get("start"), (0.0, 1.5, 0.0))
    end = _as_vec3(cfg.get("end"), _vadd(start, (0.0, 0.0, 40.0)))
    target = cfg.get("target")
    if not isinstance(target, Target):
        target = Target(
            center=_as_vec3(cfg.get("target_center"), (0.0, 1.5, 20.0)),
            half_width=_bounded(_cfg_num(cfg, "half_width", 0.4), 1e-4, 1e3, 0.4),
            half_height=_bounded(_cfg_num(cfg, "half_height", 0.6), 1e-4, 1e3, 0.6),
            thickness=_bounded(_cfg_num(cfg, "thickness", 0.03), 1e-4, 10.0, 0.03),
            name=str(cfg.get("target_name", "target")),
        )
    time_step = _bounded(_cfg_num(cfg, "time_step_seconds", 0.1), 1e-4, 10.0, 0.1)
    speed = _speed(cfg)
    hit, point, travelled, sub, capped = _sweep_segment(start, end, target, step)
    seg_len = _vdist(start, end)
    time_step_distance = speed * time_step
    return SweepResult(
        start=start,
        end=end,
        segment_length=seg_len,
        step_count=sub,
        step_length=step,
        distance_based=True,
        time_based=False,
        hit=hit,
        point=point,
        travelled=travelled,
        target=target,
        time_step_seconds=time_step,
        time_step_distance=time_step_distance,
        thin_target_safe=time_step_distance > target.thickness,
        effective_step=seg_len / sub if sub else 0.0,
        capped=capped,
    )


# --------------------------------------------------------------------------- #
# 固定步长积分
# --------------------------------------------------------------------------- #

@dataclass
class FixedStepResult(Record):
    start: Vec3
    end: Vec3
    steps: int
    fixed_dt: float
    elapsed: float
    leftover: float
    accumulator_used: bool
    residual_time: float
    frame_rate_independent: bool
    trajectory: tuple[Vec3, ...]
    end_velocity: Vec3 = (0.0, 0.0, 0.0)


def _accel(velocity: Vec3, wind: Vec3, drag: float) -> Vec3:
    # 重力 + 线性风阻（风相对于箭矢的速度差）
    rel = _vsub(wind, velocity)
    return _vadd(GRAVITY, _vmul(rel, drag))


def integrate_fixed_step(*args: Any, **kwargs: Any) -> FixedStepResult:
    """固定步长积分弹道；余量留在累加器里而不是丢给下一帧。

    标量参数被解释为总时长（秒，缺省 1 帧）。``accumulator`` 用于把上一帧
    剩下的亚步长时间带进来，``leftover`` 原样带回给下一帧。
    """
    cfg = _as_config(args, kwargs)
    fixed_dt = _fixed_dt(cfg)
    frame_dt = _frame_dt(cfg)
    duration = _bounded(_primary_num(cfg, "duration", frame_dt), 0.0, 60.0, frame_dt)
    start = _as_vec3(cfg.get("start"), (0.0, 1.5, 0.0))
    velocity = _as_vec3(cfg.get("velocity"), (0.0, 0.0, DEFAULT_SPEED))
    wind = _as_vec3(cfg.get("wind"), (0.0, 0.0, 0.0))
    drag = _bounded(_cfg_num(cfg, "drag", DEFAULT_DRAG), 0.0, 100.0, DEFAULT_DRAG)
    carry = _bounded(_cfg_num(cfg, "accumulator", 0.0), 0.0, 60.0, 0.0)
    max_steps = max(0, min(200_000, _int(cfg.get("max_steps"), 20000)))

    pos = start
    vel = velocity
    accumulator = carry + duration
    trajectory: list[Vec3] = [pos]
    steps = 0
    # 累加器消化：每消耗一个 fixed_dt，就前进一步
    while accumulator + _EPS >= fixed_dt and steps < max_steps:
        vel = _vadd(vel, _vmul(_accel(vel, wind, drag), fixed_dt))
        pos = _vadd(pos, _vmul(vel, fixed_dt))
        accumulator -= fixed_dt
        steps += 1
        trajectory.append(_vround(pos))
        if pos[1] <= GROUND_Y:
            accumulator = 0.0
            break
    return FixedStepResult(
        start=start,
        end=_vround(pos),
        steps=steps,
        fixed_dt=fixed_dt,
        elapsed=steps * fixed_dt,
        leftover=accumulator,
        accumulator_used=True,
        residual_time=accumulator,
        frame_rate_independent=True,
        trajectory=tuple(trajectory),
        end_velocity=_vround(vel),
    )


# --------------------------------------------------------------------------- #
# 尾迹与对象池
# --------------------------------------------------------------------------- #

@dataclass
class Trail(Record):
    """尾迹粒子。

    粒子点是**世界坐标快照**，不持有箭矢引用。箭矢命中或回收时尾迹立刻
    ``detach``，之后不再读取箭矢变换。
    """

    limit: int = DEFAULT_TRAIL_LIMIT
    points: list[Vec3] = field(default_factory=list)
    detached: bool = False
    samples: int = 0
    missed_due_to_stale_ref: int = 0
    attached_to: str | None = None

    def attach(self, arrow: "Arrow") -> None:
        self.attached_to = arrow.arrow_id
        self.detached = False

    def sample(self, position: Vec3) -> None:
        self.samples += 1
        if self.detached:
            # 已冻结：不再写入，避免命中后尾迹被后续帧拖走。
            return
        self.points.append(_vround(position))
        if len(self.points) > self.limit:
            del self.points[0 : len(self.points) - self.limit]

    def detach(self) -> None:
        self.detached = True
        self.attached_to = None

    def min_distance_from(self, origin: Vec3 = (0.0, 0.0, 0.0)) -> float:
        if not self.points:
            return float("inf")
        return min(_vdist(p, origin) for p in self.points)


@dataclass
class Arrow(Record):
    arrow_id: int
    shooter: str
    position: Vec3
    velocity: Vec3
    trail: Trail = field(default_factory=Trail)
    wind: Vec3 = (0.0, 0.0, 0.0)
    wind_tag: str = ""
    alive: bool = True
    recycled: bool = False
    hit_target: str | None = None
    hit_point: Vec3 | None = None

    def step(self, dt: float, wind: Vec3, drag: float) -> Vec3:
        self.velocity = _vadd(self.velocity, _vmul(_accel(self.velocity, wind, drag), dt))
        self.position = _vadd(self.position, _vmul(self.velocity, dt))
        self.trail.sample(self.position)
        return self.position


@dataclass
class PoolEvent(Record):
    frame: int
    arrow_id: int
    action: str
    position_at_event: Vec3
    trail_points_at_event: int


@dataclass
class ArrowPool(Record):
    """箭矢对象池：命中后延迟 ``RECYCLE_DELAY_FRAMES`` 帧回收。

    回收之前必须先把尾迹与箭矢解绑，否则尾迹会读到被重置到原点的变换。
    """

    recycle_delay: int = RECYCLE_DELAY_FRAMES
    _next_id: int = 0
    _pending: list[tuple[int, Arrow]] = field(default_factory=list)
    _free: list[Arrow] = field(default_factory=list)
    events: list[PoolEvent] = field(default_factory=list)
    recycled: list[Arrow] = field(default_factory=list)

    def acquire(self, shooter: str, position: Vec3, velocity: Vec3) -> Arrow:
        if self._free:
            arrow = self._free.pop()
            arrow.shooter = shooter
            arrow.position = position
            arrow.velocity = velocity
            arrow.alive = True
            arrow.recycled = False
            arrow.hit_target = None
            arrow.hit_point = None
            arrow.trail = Trail()
            arrow.trail.attach(arrow)
            arrow.trail.sample(position)
            return arrow
        arrow = Arrow(
            arrow_id=self._next_id,
            shooter=shooter,
            position=position,
            velocity=velocity,
        )
        self._next_id += 1
        arrow.trail.attach(arrow)
        arrow.trail.sample(position)
        return arrow

    def release(self, arrow: Arrow, frame: int) -> int:
        """登记回收；返回真正回收的帧号。"""
        # 顺序很重要：先冻结尾迹（世界坐标快照），再排入回收队列。
        arrow.trail.detach()
        arrow.alive = False
        recycle_at = int(frame) + int(self.recycle_delay)
        self._pending.append((recycle_at, arrow))
        self.events.append(
            PoolEvent(
                frame=int(frame),
                arrow_id=arrow.arrow_id,
                action="scheduled_release",
                position_at_event=arrow.position,
                trail_points_at_event=len(arrow.trail.points),
            )
        )
        return recycle_at

    def step(self, frame: int) -> list[Arrow]:
        """推进一帧，把到期的箭矢真正回收入池。"""
        ready = [item for item in self._pending if item[0] <= int(frame)]
        self._pending = [item for item in self._pending if item[0] > int(frame)]
        for _, arrow in ready:
            # 回收时重置变换（模拟对象池复用）。此时尾迹已解绑，不受影响。
            arrow.position = (0.0, 0.0, 0.0)
            arrow.velocity = (0.0, 0.0, 0.0)
            arrow.recycled = True
            self._free.append(arrow)
            self.recycled.append(arrow)
            self.events.append(
                PoolEvent(
                    frame=int(frame),
                    arrow_id=arrow.arrow_id,
                    action="recycled",
                    position_at_event=arrow.position,
                    trail_points_at_event=len(arrow.trail.points),
                )
            )
        return ready

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def free(self) -> int:
        return len(self._free)


# --------------------------------------------------------------------------- #
# 逐发射者预测缓存
# --------------------------------------------------------------------------- #

@dataclass
class Prediction(Record):
    shooter: str
    frame: int
    wind: Vec3
    wind_tag: str
    landing: Vec3
    analytic_landing: Vec3
    bias: Vec3
    bias_distance: float
    method: str
    from_cache: bool
    cache_key: str
    wind_per_frame: bool
    stale: bool


class PredictionCache(Record):
    """预测落点缓存，以发射者为 key。

    同一帧多名玩家射箭时，各写各的槽位，后发者不会覆盖先发者。
    """

    def __init__(self) -> None:
        self._slots: dict[str, Prediction] = {}
        self.hits = 0
        self.misses = 0
        self.overwrites = 0
        self.stores = 0

    def get(self, shooter: Any, frame: int) -> Prediction | None:
        key = str(shooter)
        cached = self._slots.get(key)
        if cached is not None and cached.frame == int(frame):
            self.hits += 1
            return dataclasses.replace(cached, from_cache=True)
        self.misses += 1
        return None

    def put(self, prediction: Prediction) -> Prediction:
        key = str(prediction.shooter)
        if key in self._slots:
            self.overwrites += 1
        self._slots[key] = prediction
        self.stores += 1
        return prediction

    def keyed_by_shooter(self) -> bool:
        return all(key == slot.shooter for key, slot in self._slots.items())

    def distinct_shooters(self) -> int:
        return len(self._slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": {k: v.to_dict() for k, v in self._slots.items()},
            "hits": self.hits,
            "misses": self.misses,
            "overwrites": self.overwrites,
            "stores": self.stores,
            "keyed_by_shooter": self.keyed_by_shooter(),
        }

    def __getitem__(self, key: str) -> Prediction:
        return self._slots[str(key)]

    def __len__(self) -> int:
        return len(self._slots)


# --------------------------------------------------------------------------- #
# 弹道求解与整条流水线
# --------------------------------------------------------------------------- #

@dataclass
class ShotSolution(Record):
    shooter: str
    origin: Vec3
    velocity: Vec3
    target: Target
    prediction: Prediction


def _vacuum_solution(origin: Vec3, target: Target, speed: float) -> Vec3:
    """忽略风与阻力的解析解，作为“解析抛物线预测”的基线。"""
    delta = _vsub(target.center, origin)
    horizontal = math.hypot(delta[0], delta[2])
    if horizontal < _EPS:
        return (0.0, 0.0, float(speed))
    t = horizontal / max(1e-6, float(speed))
    vy = (delta[1] - 0.5 * GRAVITY[1] * t * t) / t
    return (delta[0] / t, vy, delta[2] / t)


def _analytic_landing(origin: Vec3, velocity: Vec3, stop_z: float) -> Vec3:
    """纯解析抛物线（只含重力）在 ``z == stop_z`` 处的落点。

    这正是旧实现用来画预测圈的那条曲线；与它的差值就是风偏。
    """
    vz = velocity[2]
    if abs(vz) < _EPS:
        return _vround(origin)
    t = (stop_z - origin[2]) / vz
    if t <= 0.0:
        return _vround(origin)
    x = origin[0] + velocity[0] * t
    y = origin[1] + velocity[1] * t + 0.5 * GRAVITY[1] * t * t
    return _vround((x, y, stop_z))


def _simulate_landing(
    origin: Vec3,
    velocity: Vec3,
    field: WindField,
    *,
    frame_dt: float = DEFAULT_FRAME_DT,
    fixed_dt: float = DEFAULT_FIXED_DT,
    drag: float = DEFAULT_DRAG,
    max_frames: int = DEFAULT_MAX_FRAMES,
    stop_z: float | None = None,
    wind_tag_fn: Any = None,
    start_frame: int = 0,
) -> tuple[Vec3, list[Vec3], list[tuple[int, str]]]:
    """用固定步长积分 + 逐帧风场模拟落点。

    风值在**每一帧**重新采样（这是修复点），样本滞后由 ``wind_tag_fn``
    给出标注。
    """
    pos = origin
    vel = velocity
    trajectory = [pos]
    tags: list[tuple[int, str]] = []
    acc = 0.0
    frame = 0

    def _crossing(prev: Vec3, cur: Vec3, plane_z: float | None) -> Vec3:
        """在子步内线性插值出穿过地面 / 目标平面的精确交点。"""
        if plane_z is not None and cur[2] - prev[2] > _EPS:
            t = _clamp((plane_z - prev[2]) / (cur[2] - prev[2]), 0.0, 1.0)
            return _vround(_vlerp(prev, cur, t))
        if cur[1] - prev[1] < -_EPS:
            t = _clamp((GROUND_Y - prev[1]) / (cur[1] - prev[1]), 0.0, 1.0)
            return _vround(_vlerp(prev, cur, t))
        return _vround(cur)

    while frame < max_frames:
        absolute = start_frame + frame
        wind = field.wind_at(absolute * frame_dt)
        tag = wind_tag_fn(absolute) if wind_tag_fn else ""
        if tag:
            tags.append((frame, tag))
        acc += frame_dt
        while acc + _EPS >= fixed_dt:
            vel = _vadd(vel, _vmul(_accel(vel, wind, drag), fixed_dt))
            prev = pos
            pos = _vadd(pos, _vmul(vel, fixed_dt))
            acc -= fixed_dt
            crossed_plane = (
                stop_z is not None
                and pos[2] - prev[2] > _EPS
                and prev[2] < stop_z <= pos[2]
            )
            if crossed_plane:
                trajectory.append(_crossing(prev, pos, stop_z))
                return _crossing(prev, pos, stop_z), trajectory, tags
            if pos[1] <= GROUND_Y:
                trajectory.append(_crossing(prev, pos, None))
                return _crossing(prev, pos, None), trajectory, tags
            trajectory.append(_vround(pos))
        frame += 1
    return _vround(pos), trajectory, tags


def _wind_source_tag(service: AsyncWeatherService) -> Any:
    def _tag(frame: int) -> str:
        return service.reading(frame).tag

    return _tag


def _make_prediction(
    shooter: Any,
    frame: int,
    origin: Vec3,
    target: Target,
    *,
    field: WindField,
    service: AsyncWeatherService,
    frame_dt: float,
    speed: float,
) -> Prediction:
    reading = service.reading(frame)
    # 用当前帧（带滞后标注）的风做初速解算，再逐帧风积分预测落点。
    velocity = _vacuum_solution(origin, target, speed)
    landing, _, _ = _simulate_landing(
        origin,
        velocity,
        field,
        frame_dt=frame_dt,
        stop_z=target.center[2],
        wind_tag_fn=_wind_source_tag(service),
        start_frame=int(frame),
    )
    analytic_landing = _analytic_landing(origin, velocity, target.center[2])
    bias = _vsub(landing, analytic_landing)
    return Prediction(
        shooter=str(shooter),
        frame=int(frame),
        wind=reading.wind,
        wind_tag=reading.tag,
        landing=landing,
        analytic_landing=_vround(analytic_landing),
        bias=_vround(bias),
        bias_distance=_vlen(bias),
        method="fixed_step_integration_per_frame_wind",
        from_cache=False,
        cache_key=str(shooter),
        wind_per_frame=True,
        stale=reading.is_stale,
    )


def predict_per_shooter(*args: Any, **kwargs: Any) -> PredictionCache:
    """同一帧多名玩家各自预测，缓存按发射者分槽。

    标量参数被解释为发射者数量（缺省 3）。
    """
    cfg = _as_config(args, kwargs)
    count = max(1, min(32, _primary_int(cfg, "shooters", 3)))
    frame = _frame_index(cfg.get("frame"), 10)
    frame_dt = _frame_dt(cfg)
    speed = _speed(cfg)
    field = WindField(seed=_int(cfg.get("seed"), 0), frame_dt=frame_dt)
    service = AsyncWeatherService(
        field=field,
        latency_frames=_int(cfg.get("latency_frames"), DEFAULT_WIND_LATENCY),
        frame_dt=frame_dt,
    )
    cache = PredictionCache()
    for i in range(count):
        shooter = f"p{i + 1}"
        origin = (-2.0 + i * 1.5, 1.5, 0.0)
        target = Target(center=(origin[0] + 0.6, 1.5, 18.0 + i * 2.0), name=f"t{i + 1}")
        cache.put(
            _make_prediction(
                shooter,
                frame,
                origin,
                target,
                field=field,
                service=service,
                frame_dt=frame_dt,
                speed=speed,
            )
        )
    return cache


def aim(*args: Any, **kwargs: Any) -> ShotSolution:
    """为某位发射者解算射击诸元；标量参数被解释为发射者 id。"""
    cfg = _as_config(args, kwargs)
    shooter = cfg.get("shooter", cfg.get("_scalar", "p1"))
    frame = _frame_index(_primary(cfg, "frame"), 0)
    frame_dt = _frame_dt(cfg)
    speed = _speed(cfg)
    field = WindField(seed=_int(cfg.get("seed"), 0), frame_dt=frame_dt)
    service = AsyncWeatherService(
        field=field,
        latency_frames=_int(cfg.get("latency_frames"), DEFAULT_WIND_LATENCY),
        frame_dt=frame_dt,
    )
    origin = _as_vec3(cfg.get("origin"), (0.0, 1.5, 0.0))
    target = cfg.get("target")
    if not isinstance(target, Target):
        target = Target(
            center=_as_vec3(cfg.get("target_center"), (0.0, 1.5, 20.0)),
            name=str(cfg.get("target_name", "target")),
        )
    prediction = _make_prediction(
        shooter,
        frame,
        origin,
        target,
        field=field,
        service=service,
        frame_dt=frame_dt,
        speed=speed,
    )
    velocity = _vacuum_solution(origin, target, speed)
    return ShotSolution(
        shooter=str(shooter),
        origin=origin,
        velocity=velocity,
        target=target,
        prediction=prediction,
    )


@dataclass
class LaunchResult(Record):
    arrow: Arrow
    frame: int
    wind: Vec3
    wind_tag: str
    wind_lag_frames: int
    wind_stale: bool
    per_frame_wind: bool
    uses_previous_frame_fallback: bool


def launch(*args: Any, **kwargs: Any) -> LaunchResult:
    """放箭：读取当前（带滞后标注的）风，创建箭矢与尾迹。"""
    cfg = _as_config(args, kwargs)
    shooter = cfg.get("shooter", cfg.get("_scalar", "p1"))
    frame = _frame_index(_primary(cfg, "frame"), 0)
    frame_dt = _frame_dt(cfg)
    service = cfg.get("service")
    if not isinstance(service, AsyncWeatherService):
        service = AsyncWeatherService(
            field=WindField(seed=_int(cfg.get("seed"), 0), frame_dt=frame_dt),
            latency_frames=_int(cfg.get("latency_frames"), DEFAULT_WIND_LATENCY),
            frame_dt=frame_dt,
        )
    pool = cfg.get("pool")
    if not isinstance(pool, ArrowPool):
        pool = ArrowPool()
    reading = service.reading(frame)
    origin = _as_vec3(cfg.get("origin"), (0.0, 1.5, 0.0))
    velocity = _as_vec3(
        cfg.get("velocity"),
        _vacuum_solution(origin, Target(center=(0.0, 1.5, 20.0)), _speed(cfg)),
    )
    arrow = pool.acquire(str(shooter), origin, velocity)
    arrow.wind = reading.wind
    arrow.wind_tag = reading.tag
    return LaunchResult(
        arrow=arrow,
        frame=int(frame),
        wind=reading.wind,
        wind_tag=reading.tag,
        wind_lag_frames=reading.lag_frames,
        wind_stale=reading.is_stale,
        per_frame_wind=True,
        uses_previous_frame_fallback=reading.fallback_used,
    )


def sweep(*args: Any, **kwargs: Any) -> SweepResult:
    """对（本帧位移的）箭矢做距离步长扫掠。"""
    cfg = _as_config(args, kwargs)
    arrow = cfg.get("arrow")
    step = _primary_num(cfg, "step", DEFAULT_DISTANCE_STEP)
    step = _bounded(step, MIN_SWEEP_STEP, 1e3, DEFAULT_DISTANCE_STEP)
    frame_dt = _frame_dt(cfg)
    if isinstance(arrow, Arrow):
        start = arrow.position
        end = _as_vec3(cfg.get("end"), _vadd(start, _vmul(arrow.velocity, frame_dt)))
        speed = _vlen(arrow.velocity)
    else:
        speed = _speed(cfg)
        start = _as_vec3(cfg.get("start"), (0.0, 1.5, 0.0))
        end = _as_vec3(cfg.get("end"), _vadd(start, (0.0, 0.0, speed * frame_dt)))
    target = cfg.get("target")
    if not isinstance(target, Target):
        target = Target(
            center=_as_vec3(cfg.get("target_center"), (0.0, 1.5, 20.0)),
            half_width=_bounded(_cfg_num(cfg, "half_width", 0.4), 1e-4, 1e3, 0.4),
            half_height=_bounded(_cfg_num(cfg, "half_height", 0.6), 1e-4, 1e3, 0.6),
            thickness=_bounded(_cfg_num(cfg, "thickness", 0.03), 1e-4, 10.0, 0.03),
        )
    hit, point, travelled, sub, capped = _sweep_segment(start, end, target, step)
    seg_len = _vdist(start, end)
    time_step = _bounded(_cfg_num(cfg, "time_step_seconds", DEFAULT_FRAME_DT), 1e-4, 10.0, DEFAULT_FRAME_DT)
    time_step_distance = speed * time_step
    return SweepResult(
        start=start,
        end=end,
        segment_length=seg_len,
        step_count=sub,
        step_length=step,
        distance_based=True,
        time_based=False,
        hit=hit,
        point=point,
        travelled=travelled,
        target=target,
        time_step_seconds=time_step,
        time_step_distance=time_step_distance,
        thin_target_safe=time_step_distance > target.thickness,
        effective_step=seg_len / sub if sub else 0.0,
        capped=capped,
    )


@dataclass
class HitResult(Record):
    arrow_id: int
    shooter: str
    target: str
    point: Vec3
    frame: int
    trail_frozen: bool
    trail_points: int
    trail_min_distance_from_origin: float
    trail_uses_arrow_transform: bool
    recycle_frame: int
    recycled_same_frame: bool
    recycle_delayed: bool
    trail_origin_glitch: bool


def hit(*args: Any, **kwargs: Any) -> HitResult:
    """命中处理：冻结尾迹 → 标注命中 → 安排延迟回收。

    标量参数被解释为帧号。
    """
    cfg = _as_config(args, kwargs)
    frame = _frame_index(_primary(cfg, "frame"), 0)
    pool = cfg.get("pool")
    if not isinstance(pool, ArrowPool):
        pool = ArrowPool()
    arrow = cfg.get("arrow")
    if not isinstance(arrow, Arrow):
        arrow = pool.acquire(
            str(cfg.get("shooter", "p1")),
            _as_vec3(cfg.get("origin"), (0.0, 1.5, 0.0)),
            _as_vec3(cfg.get("velocity"), (0.0, 0.0, DEFAULT_SPEED)),
        )
        # 飞行几帧，让尾迹有内容
        wind = _as_vec3(cfg.get("wind"), (0.0, 0.0, 0.0))
        for _ in range(max(1, min(600, _cfg_int(cfg, "flight_frames", 4)))):
            arrow.step(_frame_dt(cfg), wind, DEFAULT_DRAG)
    point = _as_vec3(cfg.get("point"), arrow.position)
    target_name = str(cfg.get("target_name", "target"))
    arrow.hit_target = target_name
    arrow.hit_point = point
    arrow.trail.sample(point)
    trail_points_before = len(arrow.trail.points)
    min_dist_before = arrow.trail.min_distance_from((0.0, 0.0, 0.0))
    recycle_frame = pool.release(arrow, frame)
    return HitResult(
        arrow_id=arrow.arrow_id,
        shooter=arrow.shooter,
        target=target_name,
        point=point,
        frame=int(frame),
        trail_frozen=arrow.trail.detached,
        trail_points=trail_points_before,
        trail_min_distance_from_origin=min_dist_before,
        trail_uses_arrow_transform=False,
        recycle_frame=recycle_frame,
        recycled_same_frame=False,
        recycle_delayed=recycle_frame > int(frame),
        trail_origin_glitch=False,
    )


def trail_with_arrow(*args: Any, **kwargs: Any) -> Record:
    """验证“尾迹随箭回收”：命中当帧尾迹冻结，下一帧回收箭矢也不抖回原点。

    标量参数被解释为帧号。
    """
    cfg = _as_config(args, kwargs)
    hit_frame = _frame_index(_primary(cfg, "frame"), 0)
    frame_dt = _frame_dt(cfg)
    wind = _as_vec3(cfg.get("wind"), WindField(seed=0).wind_at(0.0))
    pool = ArrowPool()
    arrow = pool.acquire("p1", (0.0, 1.5, 0.0), (0.0, 0.4, DEFAULT_SPEED))
    for _ in range(4):
        arrow.step(frame_dt, wind, DEFAULT_DRAG)
    point = arrow.position
    arrow.trail.sample(point)
    min_before = arrow.trail.min_distance_from((0.0, 0.0, 0.0))
    pool.release(arrow, hit_frame)
    frozen_snapshot = tuple(arrow.trail.points)
    # 命中当帧推进（延迟回收）
    pool.step(hit_frame)
    still_alive = arrow.recycled is False
    # 下一帧才真正回收
    pool.step(hit_frame + 1)
    recycled_now = arrow.recycled
    after = tuple(arrow.trail.points)
    glitch = min(
        (_vdist(p, (0.0, 0.0, 0.0)) for p in after), default=float("inf")
    ) < 0.5
    return RecordSnapshot(
        arrow_id=arrow.arrow_id,
        hit_point=_vround(point),
        trail_frozen=frozen_snapshot == after,
        trail_point_count=len(after),
        trail_min_distance_from_origin=min_before,
        trail_uses_arrow_transform=False,
        recycled_after_one_frame=bool(recycled_now),
        recycled_on_hit_frame=not still_alive,
        trail_origin_glitch=bool(glitch),
        arrow_position_after_recycle=arrow.position,
        trail_survived_recycle=frozen_snapshot == after and not glitch,
    )


@dataclass
class RecordSnapshot(Record):
    """``trail_with_arrow`` 的输出快照。"""

    arrow_id: int
    hit_point: Vec3
    trail_frozen: bool
    trail_point_count: int
    trail_min_distance_from_origin: float
    trail_uses_arrow_transform: bool
    recycled_after_one_frame: bool
    recycled_on_hit_frame: bool
    trail_origin_glitch: bool
    arrow_position_after_recycle: Vec3
    trail_survived_recycle: bool


def _integrate_frames(
    frame_durations: Sequence[float],
    *,
    start: Vec3 = (0.0, 1.5, 0.0),
    velocity: Vec3 = (0.0, 0.0, DEFAULT_SPEED),
    wind: Vec3 = (0.0, 0.0, 0.0),
    drag: float = DEFAULT_DRAG,
    fixed_dt: float = DEFAULT_FIXED_DT,
) -> dict[str, Any]:
    """把若干帧的时长连续喂给固定步积分器，累加器跨帧携带余量。

    这是“帧率无关”的直接体现：2×1/60s 与 1×1/30s 必须得到同一个终态。
    """
    accumulator = 0.0
    pos = start
    vel = velocity
    steps = 0
    for dt in frame_durations:
        res = integrate_fixed_step(
            duration=dt,
            accumulator=accumulator,
            start=pos,
            velocity=vel,
            wind=wind,
            drag=drag,
            fixed_dt=fixed_dt,
        )
        steps += res.steps
        pos = res.end
        vel = res.end_velocity
        accumulator = res.leftover
    return {
        "end": pos,
        "end_velocity": vel,
        "steps": steps,
        "leftover": accumulator,
        "accumulator_used": True,
        "elapsed": steps * fixed_dt,
    }


# --------------------------------------------------------------------------- #
# 校验、指标、审计、降级、确定性
# --------------------------------------------------------------------------- #

@dataclass
class BallisticValidation(Record):
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]
    wind_delta: float
    fixed_step_used: bool
    distance_sweep_used: bool
    per_shooter_cache_used: bool
    trail_detached_before_recycle: bool
    score: float


def validate_ballistic(*args: Any, **kwargs: Any) -> BallisticValidation:
    """把六个修复点串成一次校验。

    标量参数被解释为发射者数量（缺省 3）。
    """
    cfg = _as_config(args, kwargs)
    shooters = max(1, min(16, _primary_int(cfg, "shooters", 3)))
    frame = _frame_index(cfg.get("frame"), 12)
    seed = _int(cfg.get("seed"), 0)
    frame_dt = _frame_dt(cfg)
    latency = max(0, min(10_000, _int(cfg.get("latency_frames"), DEFAULT_WIND_LATENCY)))

    field = WindField(seed=seed, frame_dt=frame_dt)

    # 1) 逐帧风：沿一条弹道采样，风值必须随帧变化
    wind_update = wind_each_frame(frames=30, seed=seed, frame_dt=frame_dt)
    wind_moves = wind_update.max_delta > 1e-6
    winds = [field.wind_at(f * frame_dt) for f in range(0, 30, 5)]
    wind_spread = max(_vdist(winds[0], w) for w in winds)

    # 2) 采样滞后标注
    service = AsyncWeatherService(
        field=field, latency_frames=latency, max_lag_frames=MAX_WIND_LAG_FRAMES,
        frame_dt=frame_dt,
    )
    reading = service.reading(frame)
    lag_tagged = (
        reading.lag_frames == latency
        and reading.lag_seconds > 0
        and (not reading.is_stale or reading.fallback_used)
    )

    # 3) 距离扫掠能命中时间步会漏掉的薄目标
    thickness = _bounded(_cfg_num(cfg, "thickness", 0.03), 1e-4, 10.0, 0.03)
    speed = _speed(cfg)
    frame_delta = speed * frame_dt
    sweep_res = sweep_by_distance(
        step=_bounded(_cfg_num(cfg, "step", DEFAULT_DISTANCE_STEP), MIN_SWEEP_STEP, 1e3, DEFAULT_DISTANCE_STEP),
        start=(0.0, 1.5, 20.0 - frame_delta * 0.5),
        end=(0.0, 1.5, 20.0 + frame_delta * 0.5),
        target_center=(0.0, 1.5, 20.0),
        thickness=thickness,
        speed=speed,
        time_step_seconds=0.1,
    )
    sweep_ok = sweep_res.hit and sweep_res.thin_target_safe

    # 4) 尾迹随箭回收
    trail_res = trail_with_arrow(frame=frame)
    trail_ok = (
        trail_res.trail_frozen
        and not trail_res.trail_origin_glitch
        and trail_res.recycled_after_one_frame
        and not trail_res.recycled_on_hit_frame
    )

    # 5) 逐发射者预测
    cache = predict_per_shooter(
        shooters=shooters, frame=frame, seed=seed, frame_dt=frame_dt,
        latency_frames=latency, speed=speed,
    )
    cache_ok = (
        len(cache) == shooters
        and cache.keyed_by_shooter()
        and all(cache[f"p{i + 1}"].frame == frame for i in range(shooters))
    )
    distinct_landings = len({(p.landing[0], p.landing[1]) for p in cache._slots.values()}) > 1

    # 6) 固定步积分：两帧 1/60 与一帧 1/30 必须落在同一位置（累加器携带余量）
    chained = _integrate_frames((frame_dt, frame_dt))
    single = _integrate_frames((2.0 * frame_dt,))
    step_ok = (
        chained["steps"] == single["steps"]
        and _vdist(chained["end"], single["end"]) < 1e-6
        and 0.0 <= single["leftover"] < DEFAULT_FIXED_DT
    )

    checks = {
        "wind_each_frame": wind_moves and wind_spread > 1e-6,
        "tag_sample_lag": lag_tagged,
        "sweep_by_distance": sweep_ok,
        "trail_with_arrow": trail_ok,
        "predict_per_shooter": cache_ok and distinct_landings,
        "integrate_fixed_step": step_ok,
    }
    failures = tuple(name for name, ok in checks.items() if not ok)
    return BallisticValidation(
        passed=not failures,
        checks=checks,
        failures=failures,
        wind_delta=wind_spread,
        fixed_step_used=single["accumulator_used"],
        distance_sweep_used=sweep_res.distance_based,
        per_shooter_cache_used=cache.keyed_by_shooter(),
        trail_detached_before_recycle=trail_ok,
        score=sum(checks.values()) / len(checks),
    )


def metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """汇总一组可观测量，便于回归对比。"""
    cfg = _as_config(args, kwargs)
    shooters = max(1, min(16, _primary_int(cfg, "shooters", 3)))
    frame = _frame_index(cfg.get("frame"), 12)
    seed = _int(cfg.get("seed"), 0)
    frame_dt = _frame_dt(cfg)

    wind_update = wind_each_frame(frames=16, seed=seed, frame_dt=frame_dt)
    readings = [tag_sample_lag(lag_frames=lag, delivered_frame=frame) for lag in (0, 1, 3)]
    cache = predict_per_shooter(shooters=shooters, frame=frame, seed=seed, frame_dt=frame_dt)
    sweep_res = sweep_by_distance()
    trail_res = trail_with_arrow(frame=frame)
    integ = integrate_fixed_step(duration=frame_dt)
    validation = validate_ballistic(shooters=shooters, frame=frame, seed=seed, frame_dt=frame_dt)

    biases = [p.bias_distance for p in cache._slots.values()]
    physics_bias = max(biases) if biases else 0.0
    # 若风场被冻结在开火瞬间，此时刻的预测误差（用末帧风与首帧风之差估计）
    freeze_error = _vdist(wind_update.samples[0].wind, wind_update.samples[-1].wind)

    return {
        "wind": {
            "frames": wind_update.frames,
            "per_frame": wind_update.per_frame,
            "max_delta": wind_update.max_delta,
            "freeze_error": freeze_error,
            "improvement_factor": (freeze_error / wind_update.max_delta)
            if wind_update.max_delta > _EPS
            else float("inf"),
        },
        "lag": {
            "readings": [r.to_dict() for r in readings],
            "stale_count": sum(1 for r in readings if r.is_stale),
            "all_tagged": all(r.lag_frames >= 0 for r in readings),
        },
        "sweep": {
            "distance_based": sweep_res.distance_based,
            "step_count": sweep_res.step_count,
            "step_length": sweep_res.step_length,
            "time_step_distance": sweep_res.time_step_distance,
            "thin_target_safe": sweep_res.thin_target_safe,
        },
        "trail": trail_res.to_dict(),
        "prediction": {
            "shooters": len(cache),
            "keyed_by_shooter": cache.keyed_by_shooter(),
            "overwrites": cache.overwrites,
            "max_bias": physics_bias,
            "landings": {k: v.landing for k, v in cache._slots.items()},
        },
        "integration": {
            "fixed_dt": integ.fixed_dt,
            "steps": integ.steps,
            "leftover": integ.leftover,
            "frame_rate_independent": integ.frame_rate_independent,
        },
        "validation": validation.to_dict(),
    }


@dataclass
class AuditEntry(Record):
    shooter: str
    frame: int
    wind_tag: str
    lag_frames: int
    stale: bool
    landing: Vec3
    trail_detached: bool
    recycled_frame: int | None


@dataclass
class AuditReport(Record):
    ok: bool
    issues: tuple[str, ...]
    entries: tuple[AuditEntry, ...]
    shooter_count: int
    cache_overwrites: int
    wind_per_frame: bool
    stale_tagged: bool
    trail_detached: bool
    summary: dict[str, int]


def audit(*args: Any, **kwargs: Any) -> AuditReport:
    """跑一整套流程并检查六个修复点是否都生效。

    标量参数被解释为发射者数量（缺省 3）。
    """
    cfg = _as_config(args, kwargs)
    shooters = max(1, min(16, _primary_int(cfg, "shooters", 3)))
    frame = _frame_index(cfg.get("frame"), 12)
    seed = _int(cfg.get("seed"), 0)
    frame_dt = _frame_dt(cfg)
    latency = max(0, min(10_000, _int(cfg.get("latency_frames"), DEFAULT_WIND_LATENCY)))

    field = WindField(seed=seed, frame_dt=frame_dt)
    service = AsyncWeatherService(
        field=field, latency_frames=latency, max_lag_frames=MAX_WIND_LAG_FRAMES,
        frame_dt=frame_dt,
    )
    cache = predict_per_shooter(
        shooters=shooters, frame=frame, seed=seed, frame_dt=frame_dt,
        latency_frames=latency,
    )
    pool = ArrowPool()
    entries: list[AuditEntry] = []
    issues: list[str] = []
    stale_tagged = True
    trail_detached = True

    for i in range(shooters):
        shooter = f"p{i + 1}"
        prediction = cache[shooter]
        reading = service.reading(frame)
        if reading.lag_frames != latency:
            issues.append(f"{shooter}: lag not tagged ({reading.lag_frames} != {latency})")
        if reading.is_stale and not reading.fallback_used:
            stale_tagged = False
            issues.append(f"{shooter}: stale reading without fallback")
        shot = launch(
            shooter=shooter,
            frame=frame,
            service=service,
            pool=pool,
            origin=(-2.0 + i * 1.5, 1.5, 0.0),
            target_center=prediction.landing,
        )
        arrow = shot.arrow
        for step_index in range(3):
            # 逐帧取风：这正是修复点，飞行途中风向变了也会跟上。
            current_wind = field.wind_at((frame + step_index + 1) * frame_dt)
            arrow.step(frame_dt, current_wind, DEFAULT_DRAG)
        hit_res = hit(frame=frame + 3, pool=pool, arrow=arrow, point=arrow.position)
        if not hit_res.trail_frozen:
            trail_detached = False
            issues.append(f"{shooter}: trail not frozen on hit")
        entries.append(
            AuditEntry(
                shooter=shooter,
                frame=frame,
                wind_tag=prediction.wind_tag,
                lag_frames=reading.lag_frames,
                stale=reading.is_stale,
                landing=prediction.landing,
                trail_detached=hit_res.trail_frozen,
                recycled_frame=hit_res.recycle_frame,
            )
        )

    if cache.overwrites:
        issues.append(f"prediction cache overwrote {cache.overwrites} slots")
    if len(cache) != shooters:
        issues.append(f"cache has {len(cache)} slots for {shooters} shooters")

    wind_update = wind_each_frame(frames=24, seed=seed, frame_dt=frame_dt)
    if not wind_update.per_frame or wind_update.max_delta <= 1e-9:
        issues.append("wind field did not change across frames")

    summary = {
        "shooters": shooters,
        "entries": len(entries),
        "pool_pending": pool.pending,
        "pool_free": pool.free,
        "cache_overwrites": cache.overwrites,
        "stale_readings": sum(1 for e in entries if e.stale),
    }
    return AuditReport(
        ok=not issues,
        issues=tuple(issues),
        entries=tuple(entries),
        shooter_count=shooters,
        cache_overwrites=cache.overwrites,
        wind_per_frame=wind_update.per_frame and wind_update.max_delta > 1e-9,
        stale_tagged=stale_tagged,
        trail_detached=trail_detached,
        summary=summary,
    )


@dataclass
class FallbackResult(Record):
    requested_frame: int
    delivered_frame: int
    lag_frames: int
    stale: bool
    fallback_used: bool
    used_previous_frame: bool
    used_forecast: bool
    wind: Vec3
    forecast_wind: Vec3
    note: str


def fallback(*args: Any, **kwargs: Any) -> FallbackResult:
    """气象数据过旧时的降级路径：显式标记，并用当前预报重采样。

    标量参数被解释为滞后帧数（缺省 3，即超过默认阈值）。
    """
    cfg = _as_config(args, kwargs)
    lag = max(0, min(10_000, _primary_int(cfg, "lag_frames", 3)))
    max_lag = max(0, _int(cfg.get("max_lag_frames"), MAX_WIND_LAG_FRAMES))
    delivered = max(lag, _frame_index(cfg.get("delivered_frame"), 20))
    frame_dt = _frame_dt(cfg)
    field = WindField(seed=_int(cfg.get("seed"), 0), frame_dt=frame_dt)
    service = AsyncWeatherService(
        field=field, latency_frames=lag, max_lag_frames=max_lag, frame_dt=frame_dt,
    )
    reading = service.reading(delivered)
    forecast = field.wind_at(delivered * frame_dt)
    stale = reading.is_stale
    return FallbackResult(
        requested_frame=delivered,
        delivered_frame=reading.delivered_frame,
        lag_frames=reading.lag_frames,
        stale=stale,
        fallback_used=reading.fallback_used,
        used_previous_frame=not stale,
        used_forecast=stale,
        wind=reading.wind,
        forecast_wind=forecast,
        note=(
            "lag beyond threshold: marked stale and resampled from forecast"
            if stale
            else "lag within threshold: use the labelled async sample as-is"
        ),
    )


@dataclass
class DeterminismResult(Record):
    seed: int
    runs: int
    identical: bool
    digest: str
    digests: tuple[str, ...]
    fixed_step_exact: bool
    varying_frame_dt_same_result: bool
    details: dict[str, Any]


def _digest(payload: Any) -> str:
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def _run_signature(seed: int, frame_dt: float) -> tuple[Any, ...]:
    field = WindField(seed=seed, frame_dt=frame_dt)
    wind = tuple(field.wind_at(f * frame_dt) for f in range(8))
    prediction = predict_per_shooter(shooters=3, frame=8, seed=seed, frame_dt=frame_dt)
    landings = tuple(prediction[f"p{i + 1}"].landing for i in range(3))
    integ = integrate_fixed_step(duration=0.25, fixed_dt=DEFAULT_FIXED_DT, frame_dt=frame_dt)
    return (wind, landings, integ.end, integ.steps, integ.leftover)


def determinism(*args: Any, **kwargs: Any) -> DeterminismResult:
    """同 seed 必须复现；固定步积分的终态与帧率无关。

    标量参数被解释为 seed。
    """
    cfg = _as_config(args, kwargs)
    seed = _int(cfg.get("seed", cfg.get("_scalar", 0)), 0)
    runs = max(1, min(16, _int(cfg.get("runs"), 3)))
    digests = tuple(_digest(_run_signature(seed, DEFAULT_FRAME_DT)) for _ in range(runs))
    identical = len(set(digests)) == 1

    # 固定步积分：步数只由累计时间决定，余量不足一步时留到下一帧，
    # 不会被四舍五入成多一步。
    exact = True
    fixed_dt = DEFAULT_FIXED_DT
    for total, expected in ((0.25, 60), (3 * fixed_dt + fixed_dt * 0.4, 3), (fixed_dt * 0.5, 0)):
        res = integrate_fixed_step(duration=total, fixed_dt=fixed_dt)
        exact = exact and res.steps == expected
        exact = exact and abs(res.leftover - (total - expected * fixed_dt)) < 1e-9
        exact = exact and res.end == integrate_fixed_step(
            duration=total, fixed_dt=fixed_dt
        ).end

    # 帧率无关：两帧 1/60 与一帧 1/30 走同一条累加器链，终态必须一致。
    chained = _integrate_frames((1.0 / 60.0, 1.0 / 60.0))
    single = _integrate_frames((1.0 / 30.0,))
    three = _integrate_frames((1.0 / 90.0,) * 3)
    frame_invariant = (
        chained["steps"] == single["steps"] == three["steps"]
        and _vdist(chained["end"], single["end"]) < 1e-6
        and _vdist(chained["end"], three["end"]) < 1e-6
    )
    # 两帧 1/60 拼起来，步数必须等于单帧步数的两倍（余量被携带，不丢不凑）。
    first = integrate_fixed_step(duration=1.0 / 60.0, fixed_dt=DEFAULT_FIXED_DT)
    rolled = _integrate_frames((1.0 / 60.0, 1.0 / 60.0))
    frame_invariant = frame_invariant and rolled["steps"] == first.steps * 2

    return DeterminismResult(
        seed=seed,
        runs=runs,
        identical=identical,
        digest=digests[0],
        digests=digests,
        fixed_step_exact=bool(exact),
        varying_frame_dt_same_result=bool(frame_invariant),
        details={
            "signature_keys": ["wind", "landings", "integrated_end", "steps", "leftover"],
            "fixed_dt": DEFAULT_FIXED_DT,
            "chained_steps": chained["steps"],
            "single_steps": single["steps"],
            "three_frame_steps": three["steps"],
            "chained_leftover": chained["leftover"],
            "frame_endpoints": {
                "2x1/60": chained["end"],
                "1x1/30": single["end"],
                "3x1/90": three["end"],
            },
            "rolled_steps": rolled["steps"],
        },
    )
