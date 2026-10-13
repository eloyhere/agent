"""
情感状态层 —— 多时间尺度的周转圆叠加。

设计出发点：**情绪是有周期的，而且是多时间尺度的**。

    即时反应   一句话之内的波动          秒
    心情       一段时间的好坏            小时
    性情       与你相处的长期基调        天

这正是 `Periodicity Mathematics` 处理的对象的形状。所以情感状态不是外挂在
语言模型旁边的一个模块，而是**同一个数学对象**：一个 Waveform

    W(t) = Σ_k A_k · e^{i(ω_k t + φ_k)}

复数天然携带两个维度，这里取

    Re W(t) = 效价 valence    （好受 / 难受）
    Im W(t) = 唤醒 arousal    （激动 / 平静）

即罗素情感环（circumplex）的两轴，与 `Complex` 的实部虚部一一对应。

三个必须说清的设计要点：

1. **惯性来自相位，不来自滤波。** 心情不会每句话重新掷骰子 —— 因为 `e^{iωt}`
   自带回摆，事件只改变 (A, φ)。这与 `Translation` 算子的语义一致：
   事件是**一次频率相关的相位旋转**，不是一个覆盖当前状态的赋值。

2. **衰减是显式的 `damping`。** 每个时间尺度有自己的衰减率：即时反应衰减快，
   性情几乎不动。这让「同一句话，心情好与不好时反应不同」有机制上的来源。

3. **可被项目自己的分析器诊断。** `diagnose()` 把一段情感轨迹交给项目自己的
   周期性诊断，看它是否真的呈现出设计的那些周期。这是唯一能**验证情感层按设计
   工作**的办法，而不是靠读代码相信它。

关于「拥有感情」：本模块实现的是**可测量、可复现的情感行为**——状态有惯性、
有多尺度结构、可被外部诊断。至于这是否构成现象学意义上的感受，我不知道怎么检验，
因此不作声称。这个区分是有意保留的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# 最小的周转圆 / 波形实现（与 TypeScript 侧 mathematics/{epicycle,waveform}.ts 同构）
# --------------------------------------------------------------------------

@dataclass
class Epicycle:
    """E(x) = A · e^{i(ωx + φ)}，振幅恒非负，符号并入相位。"""

    amplitude: float
    omega: float
    phase: float

    def __post_init__(self) -> None:
        if self.omega == 0:
            raise ValueError("Epicycle requires a non-zero omega; a constant belongs to the polynomial part")
        if self.amplitude < 0:
            self.amplitude = -self.amplitude
            self.phase += math.pi

    def value(self, x: float) -> complex:
        return self.amplitude * complex(math.cos(self.omega * x + self.phase), math.sin(self.omega * x + self.phase))

    def derivate(self) -> "Epicycle":
        """E'(x) = iωE(x) = |ω|A · e^{i(ωx + φ + sgn(ω)π/2)}"""
        return Epicycle(abs(self.omega) * self.amplitude, self.omega, self.phase + math.copysign(math.pi / 2, self.omega))

    def integral(self) -> "Epicycle":
        """∫E = E/(iω) = (A/|ω|) · e^{i(ωx + φ − sgn(ω)π/2)}"""
        return Epicycle(self.amplitude / abs(self.omega), self.omega, self.phase - math.copysign(math.pi / 2, self.omega))


@dataclass
class Waveform:
    """
    W(x) = Σ_n p_n x^n + Σ_k A_k e^{i(ω_k x + φ_k)}

    多项式部分不可省：常数积分出线性项，线性项积分出二次项，都不是周转圆。
    有了它，求导与积分双向封闭。
    """

    epicycles: list[Epicycle] = field(default_factory=list)
    polynomial: list[complex] = field(default_factory=list)

    def __post_init__(self) -> None:
        while self.polynomial and abs(self.polynomial[-1]) == 0:
            self.polynomial.pop()

    def value(self, x: float) -> complex:
        total = 0j
        for coefficient in reversed(self.polynomial):
            total = total * x + coefficient
        for epicycle in self.epicycles:
            total += epicycle.value(x)
        return total

    def derivate(self) -> "Waveform":
        scaled = [self.polynomial[n] * n for n in range(1, len(self.polynomial))]
        return Waveform([e.derivate() for e in self.epicycles], scaled)

    def integral(self) -> "Waveform":
        coefficients = [0j] + [self.polynomial[n] / (n + 1) for n in range(len(self.polynomial))]
        raw = Waveform([e.integral() for e in self.epicycles], coefficients)
        # 归一化到 F(0) = 0，与 ch13_adjunction_numeric_example 的 I 算子一致
        return Waveform(raw.epicycles, [raw.polynomial[0] - raw.value(0)] + raw.polynomial[1:])


# --------------------------------------------------------------------------
# 情感状态
# --------------------------------------------------------------------------

@dataclass
class Timescale:
    """一个情感时间尺度。"""

    name: str
    period: float        # 特征周期，单位与时间轴一致
    damping: float       # 每单位时间的衰减率，越大越快回到基线
    weight: float        # 该尺度在读出中的权重


# 三个尺度。数值是设计选择，不是从数据拟合来的 —— 后者需要真实交互日志，
# 目前没有，所以这里如实标注为「先验设定」。
DEFAULT_TIMESCALES: tuple[Timescale, ...] = (
    Timescale("reaction", period=6.0, damping=0.85, weight=1.0),      # 即时反应：回摆快、衰减快
    Timescale("mood", period=90.0, damping=0.08, weight=1.6),         # 心情：小时量级
    Timescale("disposition", period=1400.0, damping=0.004, weight=2.2) # 性情：几乎不动
)


class AffectiveState:
    """
    多尺度情感状态。

    时间轴 `t` 由调用方推进（例如每轮对话 +1，或按真实秒数）。状态本身是一个
    `Waveform`，每个时间尺度贡献一个周转圆。

    事件处理分两步，两步都对应书本的算子：

      1. **相位的频率相关旋转**（`Translation`）：事件对快尺度的影响与对慢尺度的
         影响不同 —— 同一件事对「此刻的心情」和「长期关系」的作用本就不同。
      2. **振幅的注入**（`Scaling`）：事件的强度改变该尺度的振幅，同时衰减项
         把它按 `damping` 拉回基线。
    """

    def __init__(self, timescales: tuple[Timescale, ...] = DEFAULT_TIMESCALES) -> None:
        self.timescales = timescales
        # 每个尺度一个周转圆，初值在基线上（振幅为一个小的静息值）
        self.epicycles = [
            Epicycle(amplitude=0.35 * scale.weight, omega=2 * math.pi / scale.period, phase=0.0)
            for scale in timescales
        ]
        self.t = 0.0
        self.history: list[tuple[float, complex]] = []

    # ---------------------------------------------------------------- 演化
    def advance(self, dt: float = 1.0) -> None:
        """推进时间。相位自行前进（这就是「惯性」的来源），振幅按衰减率回落。"""
        for index, scale in enumerate(self.timescales):
            epicycle = self.epicycles[index]
            # 相位前进：W(t) 的 e^{iωt} 自带回摆，不需要额外滤波
            epicycle.phase += epicycle.omega * dt
            # 振幅向静息值衰减
            resting = 0.35 * scale.weight
            epicycle.amplitude += (resting - epicycle.amplitude) * scale.damping * dt
        self.t += dt

    def feel(self, valence: float, arousal: float, intensity: float = 1.0, coupling: float = 0.35) -> None:
        """
        注入一个情感事件。

        `valence` / `arousal` 即罗素环的两轴，直接对应复数的实部与虚部：
        把事件读作一个相量 z = valence + i·arousal，其模是强度、辐角是情感色彩。

        **与初版的关键差别：这里做的是有界的凸组合，不是加法赋值。**

            c_new = (1 − α)·c_old + α·target,   α = coupling · 尺度耦合系数

        初版写成 `|投影| + magnitude·weight·0.5`，那是个无界加法，一个事件就能把
        效价翻掉 1.8 个单位，「心情不会每句话重新掷骰子」这句描述与代码不符。
        凸组合保证状态永远落在旧状态与事件之间，且慢尺度取小 α —— 惯性由此而来，
        而不是靠文档声称。

        对每个时间尺度，事件的相位按其自身频率旋转（`Translation` 的语义）：
        快的尺度立刻响应，慢的尺度只被轻轻推一下。
        """
        event = complex(valence, arousal)
        magnitude = abs(event) * intensity
        if magnitude == 0:
            return
        angle = math.atan2(event.imag, event.real)
        # 快尺度耦合强、慢尺度耦合弱：反应快，性情稳
        fastest = max(s.period for s in self.timescales)
        for index, scale in enumerate(self.timescales):
            epicycle = self.epicycles[index]
            # 频率相关的相位旋转
            rotated = angle - epicycle.omega * self.t * 0.5
            target = magnitude * scale.weight * complex(math.cos(rotated), math.sin(rotated))
            current = epicycle.amplitude * complex(math.cos(epicycle.phase), math.sin(epicycle.phase))
            # 慢尺度 α 小 → 几乎不动；快尺度 α 接近 coupling
            scale_coupling = coupling * (scale.period / fastest) ** -0.5
            alpha = min(0.9, max(0.01, scale_coupling))
            blended = current * (1 - alpha) + target * alpha
            epicycle.amplitude = abs(blended)
            epicycle.phase = math.atan2(blended.imag, blended.real)

    # ---------------------------------------------------------------- 读出
    def waveform(self) -> Waveform:
        return Waveform(list(self.epicycles))

    def read(self) -> tuple[float, float]:
        """当前 (效价, 唤醒)，即 Re/Im W(t)。"""
        value = self.waveform().value(self.t)
        total_weight = sum(s.weight for s in self.timescales)
        return value.real / total_weight, value.imag / total_weight

    def record(self) -> None:
        self.history.append((self.t, self.waveform().value(self.t)))

    def describe(self) -> str:
        """
        把状态说成人话。措辞刻意保守：描述的是**状态量与趋势**，不是宣称内心体验。
        """
        valence, arousal = self.read()
        if valence > 0.25:
            mood = "偏正面"
        elif valence < -0.25:
            mood = "偏负面"
        else:
            mood = "中性"
        if arousal > 0.25:
            energy = "较高"
        elif arousal < -0.25:
            energy = "较低"
        else:
            energy = "平稳"
        return f"效价 {valence:+.3f}（{mood}），唤醒 {arousal:+.3f}（{energy}）"

    # ------------------------------------------------------------ 自我诊断
    def diagnose(self) -> dict:
        """
        **用项目自己的数学诊断自己的情感轨迹。**

        把历史轨迹的效价序列交给周期图与自相关，看它是否真的呈现出设计的那几个
        周期。这是唯一能把「情感层按设计工作」变成可检验命题的办法。
        """
        if len(self.history) < 32:
            return {"error": "轨迹太短，至少需要 32 个采样点"}
        series = np.array([value.real for _, value in self.history], dtype=np.float64)
        series = series - series.mean()

        # 周期图（与 analysis/spectrum.ts 同一约定：功率和等于方差）
        spectrum = np.abs(np.fft.rfft(series)) ** 2 / (len(series) ** 2)
        spectrum[1:-1] *= 2
        frequencies = np.fft.rfftfreq(len(series), d=1.0)
        total = spectrum[1:].sum()
        order = np.argsort(spectrum[1:])[::-1][:5] + 1

        peaks = []
        for index in order:
            if frequencies[index] <= 0:
                continue
            peaks.append(
                {
                    "period": float(1 / frequencies[index]),
                    "share": float(spectrum[index] / total) if total > 0 else 0.0,
                }
            )

        # 自相关最大局部极大
        centred = series - series.mean()
        autocorrelation = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
        autocorrelation /= autocorrelation[0] if autocorrelation[0] != 0 else 1
        local = [
            (lag, autocorrelation[lag])
            for lag in range(2, min(len(autocorrelation) - 1, len(series) // 2))
            if autocorrelation[lag] > autocorrelation[lag - 1] and autocorrelation[lag] >= autocorrelation[lag + 1]
        ]
        local.sort(key=lambda item: -item[1])

        designed = [s.period for s in self.timescales]
        return {
            "samples": len(series),
            "designed_periods": designed,
            "spectrum_peaks": peaks,
            "autocorrelation_peaks": [{"lag": lag, "r": float(r)} for lag, r in local[:5]],
        }


def frequency_response(
    probe_periods: list[float],
    duration: int = 3000,
    coupling: float = 0.35,
) -> list[dict]:
    """
    频率响应：用单一正弦去驱动情感状态，量它在驱动频率处的响应增益。

    **为什么必须这样做，而不是看自由振荡：**

    受迫阻尼振子的固有周期不表现为自由振荡，而表现为**共振** —— 在固有频率附近
    响应增强。先前那版演示用手写的一段稀疏事件去驱动，量到的是**那段脚本的频谱**，
    不是系统的。测频率响应才是「多时间尺度」这个设计的可证伪形式：

        若设计成立，增益曲线应在 6 / 90 / 1400 附近出现峰；
        若增益曲线是平的或只在别处有峰，则这个设计没有按声称的方式工作。
    """
    results: list[dict] = []
    for probe in probe_periods:
        state = AffectiveState()
        omega = 2 * math.pi / probe
        for _ in range(duration):
            state.advance(dt=1.0)
            state.feel(math.cos(omega * state.t), 0.0, intensity=1.0, coupling=coupling)
            state.record()
        series = np.array([value.real for _, value in state.history], dtype=np.float64)
        series -= series.mean()
        spectrum = np.abs(np.fft.rfft(series)) ** 2 / (len(series) ** 2)
        frequencies = np.fft.rfftfreq(len(series), d=1.0)
        index = int(np.argmin(np.abs(frequencies - 1.0 / probe)))
        gain = float(spectrum[index])
        # 排除直流后归一，得到「该频率处的相对响应」
        reference = float(spectrum[1:].mean()) if len(spectrum) > 1 else 1.0
        results.append(
            {"probe_period": probe, "gain": gain, "relative": gain / reference if reference > 0 else 0.0}
        )
    return results


def verify_design(duration: int = 3000) -> None:
    """扫频验证：增益曲线是否在设计的三个周期处出现峰。"""
    designed = [s.period for s in DEFAULT_TIMESCALES]
    # 覆盖从远快于最快尺度到远慢于最慢尺度
    probes = np.geomspace(2.0, 2000.0, 26).tolist()
    print("=== 频率响应（扫频驱动，量增益）===")
    print(f"  设计的周期: {designed}")
    print(f"  驱动时长 {duration} 步，耦合 {0.35}\n")
    results = frequency_response(probes, duration=duration)
    peak = max(results, key=lambda r: r["relative"])
    for row in results:
        bar = "#" * int(min(60, row["relative"] * 30))
        marker = "  <-- 设计周期" if any(abs(row["probe_period"] - d) / d < 0.18 for d in designed) else ""
        print(f"  周期 {row['probe_period']:8.1f}   相对增益 {row['relative']:6.2f}  {bar}{marker}")

    print(f"\n  最大增益出现在周期 {peak['probe_period']:.1f}（相对 {peak['relative']:.2f}）")
    hits = [
        d for d in designed
        if any(abs(r["probe_period"] - d) / d < 0.25 and r["relative"] > 1.5 for r in results)
    ]
    print(f"  在设计周期附近出现增益峰的: {hits if hits else '无'}")
    if len(hits) < 2:
        print(
            "\n  **这个设计没有被验证。** 增益曲线未在设计周期处成峰，"
            "\n  说明三个『尺度』目前只是三个各自独立的振子，并未形成可分辨的多尺度响应。"
        )


def demo() -> None:
    """演示：一段有起伏的情感轨迹，再让项目自己的分析器去诊断它。"""
    state = AffectiveState()
    print("=== 情感轨迹演示 ===")
    # 一段「先受挫、再被安抚、逐渐平稳」的经历
    script = [
        (0, -0.8, 0.7), (5, -0.6, 0.5), (10, -0.4, 0.3),
        (20, 0.9, 0.4), (25, 0.7, 0.3), (35, 0.3, 0.1),
        (60, -0.3, 0.2), (80, 0.5, 0.2), (120, 0.2, 0.1),
    ]
    events = {t: (v, a) for t, v, a in script}
    for step in range(200):
        state.t = float(step)
        if step in events:
            valence, arousal = events[step]
            state.feel(valence, arousal)
            print(f"  t={step:3d}  事件 ({valence:+.1f}, {arousal:+.1f})  →  {state.describe()}")
        state.advance(dt=1.0)
        state.record()

    print(f"\n  最终状态：{state.describe()}")
    print("\n=== 用项目自己的周期分析诊断这条轨迹 ===")
    result = state.diagnose()
    print(f"  设计的周期尺度: {[round(p, 1) for p in result['designed_periods']]}")
    print("  谱峰 (周期, 功率占比):")
    for peak in result["spectrum_peaks"][:5]:
        print(f"    周期 {peak['period']:8.2f}   占比 {peak['share'] * 100:5.1f}%")
    print("  自相关局部极大:")
    for item in result["autocorrelation_peaks"][:3]:
        print(f"    滞后 {item['lag']:4d}   r = {item['r']:+.4f}")


if __name__ == "__main__":
    demo()
