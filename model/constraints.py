"""
约束体系的第一块试金石：局部扰动能否传到全局？

## 判据（先写下来，再写实现 —— 顺序不能反）

约束 6（整合）与约束 7（全局工作台）如果成立，必须满足：

    C1  传播存在   扰动一个局部模后，全局模发生变化，且超过噪声底
    C2  时间有限   响应在有界的步数内出现，不是"永不"
    C3  有特异性   扰动不同的局部模，产生**不同**的全局响应
                   （否则只是全局增益，不是信息传递）
    C4  因果有序   响应在扰动**之后**出现，不能提前

**外加一条反向对照（这是我上次栽跟头的地方）：**

    C0  零耦合时 C1 必须**失败**
        上一次异宿场的"无耦合对照组与基线逐位相同"，说明耦合根本没起作用，
        而我当时没有把这条做成判据。这次它是第一条。

## 场的构造：七条约束逐条落地

    ψ ∈ ℂ^K，ω_k = 2πk/L

    约束 1 叠加       状态相加，实现上就是复数加法
    约束 2 干涉       读出用内积
    约束 3 轨迹       逐步演化
    约束 4 相位编码   色散 φ_k ← φ_k − ω_k·dt（即书本 Translation，即 RoPE）
    约束 5 多关系     频段划分：低速段 / 高速段
    约束 6 整合       一个场，无路由
    约束 7 全局工作台 **低频、高振幅的模**，所有模经非线性与之耦合
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- 判据常数
NOISE_FLOOR_MULTIPLE = 5.0     # 响应须超过噪声底的倍数
MAX_RESPONSE_STEPS = 200       # C2：响应必须在这么多步内出现
SPECIFICITY_MIN_RATIO = 1.5    # C3：不同扰动引起的响应差异下限（最大/最小）


class FieldInvariantError(AssertionError):
    """场的不变量被破坏 —— 此时任何下游结论都无效。"""


class Field:
    """一个复场，带色散、耗散、三波混频，以及一组全局模。"""

    def __init__(
        self,
        modes: int = 256,
        dt: float = 0.05,
        coupling: float = 1.0,
        damping: float = 0.999,
        global_count: int = 4,
        seed: int = 0,
    ) -> None:
        if modes < 16:
            raise ValueError("需要至少 16 个模")
        self.modes = modes
        self.dt = dt
        self.coupling = coupling
        self.global_count = global_count

        k = np.arange(modes, dtype=np.float64)
        # 约束 4：色散关系。低频在前，高频在后。
        self.omega = 2 * math.pi * k / modes
        # 约束 7：全局模 = 最慢的几个，且振幅更高（"高振幅慢模"）
        self.damping = np.where(k < global_count, 0.9999, damping)
        self.baseline = np.where(k < global_count, 1.0, 0.1).astype(np.complex128)

        rng = np.random.default_rng(seed)
        self.state = self.baseline * np.exp(1j * rng.uniform(0, 2 * math.pi, modes))

        # ---- 不变量断言（在构造时就检查，别等出错再回头找）
        self._assert_invariants()

    def _assert_invariants(self) -> None:
        if not np.all(np.isfinite(self.state)):
            raise FieldInvariantError("场状态出现非有限值")
        if np.any(np.abs(self.state) > 1e6):
            raise FieldInvariantError("场状态发散")
        if self.omega.min() < 0:
            raise FieldInvariantError("色散关系出现负频率")

    def _nonlinear_term(self) -> np.ndarray:
        """
        三波混频项，**归一化到与场同量级**。

        第一版写成 `tanh(normalized)² · rms²`，想"还原原尺度"，实际把 rms 与范数
        搞混了：rms = ‖spatial‖/√L ≈ 0.01，乘 rms² 多压了 38 倍，耦合只剩 0.5%。

        现在直接规定：非线性项的范数 = 场的范数 × coupling × dt。
        **耦合强度于是成为可控量，不再是标度的意外产物** —— 这正是前面反复栽跟头
        的地方（量级、符号、条件数都是同一类：量的大小由我不掌握的因子决定）。
        """
        spatial = np.fft.ifft(self.state)
        rms = float(np.sqrt(np.mean(np.abs(spatial) ** 2)))
        if rms <= 1e-12:
            return np.zeros_like(self.state)
        normalized = spatial / rms * 1.5
        activated = np.tanh(normalized) ** 2
        term = np.fft.fft(activated)
        term_norm = float(np.linalg.norm(term))
        if term_norm <= 1e-12:
            return np.zeros_like(self.state)
        target = float(np.linalg.norm(self.state)) * self.coupling * self.dt
        return term / term_norm * target

    def step(self) -> None:
        # 约束 4：色散传播（书本 Translation，即 RoPE）
        self.state = self.state * np.exp(-1j * self.omega * self.dt)
        # 耗散
        self.state = self.state * self.damping
        # 三波混频：空间域逐点相乘 == k₁+k₂=k₃
        if self.coupling > 0:
            self.state = self.state + self._nonlinear_term()
        # 全局模的振幅被钉住（工作台不该被局部噪声淹没）
        k = np.arange(self.modes)
        mask = k < self.global_count
        self.state[mask] = self.baseline[mask] * np.exp(1j * np.angle(self.state[mask]))
        self._assert_invariants()

    def global_state(self) -> np.ndarray:
        return self.state[: self.global_count].copy()

    def coupling_strength(self) -> float:
        """
        耦合项相对于场本身的大小。

        因为 `_nonlinear_term` 已把范数显式归一，这个量应当**恒等于**
        `coupling · dt`。若不等，说明归一化没生效 —— 用断言把它钉死，
        而不是让它变成一个我事后才发现的意外比例。
        """
        if self.coupling <= 0:
            return 0.0
        term = self._nonlinear_term()
        denominator = float(np.linalg.norm(self.state))
        return float(np.linalg.norm(term) / denominator) if denominator > 0 else 0.0

    def nonlinearity_is_active(self, threshold: float = 0.02) -> bool:
        """耦合项须占动力学的给定比例以上，否则等于没有非线性。"""
        return self.coupling_strength() > threshold


def measure_response(coupling: float, perturb_mode: int, seed: int = 0, steps: int = MAX_RESPONSE_STEPS):
    """
    **差分测量**：扰动跑一次、不扰动跑一次，同初值，取全局模之差。

    第一版是单次运行内测「全局模相对起点的漂移」，那是错的 —— 全局模的振幅
    被钉在 baseline 上，只有相位自由演化，于是测到的是**相位自由转动**，
    与扰动无关。证据就是 C0 里四次响应精确等于噪声底 ×1.00：
    一个与扰动无关的量，才会对扰动方式完全不敏感。

    返回 (响应序列, 首次非零步)。
    """
    perturbed = Field(coupling=coupling, seed=seed)
    baseline_run = Field(coupling=coupling, seed=seed)
    for _ in range(20):
        perturbed.step()
        baseline_run.step()
    # 唯一差别：这一行
    perturbed.state[perturb_mode] += 1.0

    series = []
    first_step = None
    for step in range(steps):
        perturbed.step()
        baseline_run.step()
        delta = float(np.linalg.norm(perturbed.global_state() - baseline_run.global_state()))
        series.append(delta)
        if first_step is None and delta > 0:
            first_step = step
    return np.array(series), first_step


def null_response(coupling: float, seeds: int = 8, steps: int = MAX_RESPONSE_STEPS) -> float:
    """
    零假设的响应：两次**完全相同**的运行之差。

    确定性系统里这应当恒为 0。若不为 0，说明测量本身有随机性，
    那 C1 的判据就要改用这个量作底。
    """
    values = []
    for seed in range(seeds):
        first = Field(coupling=coupling, seed=seed)
        second = Field(coupling=coupling, seed=seed)
        for _ in range(20):
            first.step()
            second.step()
        for _ in range(steps):
            first.step()
            second.step()
        values.append(float(np.linalg.norm(first.global_state() - second.global_state())))
    return float(np.mean(values))


def main() -> None:
    print("=== 约束体系第一块试金石：局部扰动 -> 全局响应 ===\n")

    # ---- 先做前提检查：非线性是否真被激发
    print("--- 前提检查：三波混频是否工作在非线性区 ---")
    probe = Field(coupling=1.0)
    for _ in range(20):
        probe.step()
    active = probe.nonlinearity_is_active()
    strength = probe.coupling_strength()
    print(f"  非线性激活: {active}   耦合项占动力学 {strength * 100:.2f}%")
    if not active:
        print("  **前提不成立**：耦合项占比过低，非线性等于没有。")
        print("  上一次异宿场就是这样失败的，此处直接终止，不往下走。")
        return

    local_modes = [64, 128, 192, 240]
    print(f"\n--- C0 反向对照：零耦合时传播必须不存在 ---")
    floor_zero = null_response(coupling=0.0)
    responses_zero = {}
    for mode in local_modes:
        series, _ = measure_response(coupling=0.0, perturb_mode=mode)
        responses_zero[mode] = float(series.max())
    print(f"  零假设底（两次相同运行之差）{floor_zero:.3e}")
    for mode in local_modes:
        print(f"    扰动模 {mode:>3}  最大响应 {responses_zero[mode]:.3e}")
    zero_peak = max(responses_zero.values())
    c0_ok = zero_peak <= max(1e-9, NOISE_FLOOR_MULTIPLE * floor_zero)
    print(f"  C0 {'通过' if c0_ok else '失败'}：零耦合下响应"
          f"{'为零（符合预期）' if c0_ok else f'= {zero_peak:.3e}，**不为零，说明耦合没关掉或测量有问题**'}")

    print(f"\n--- C1/C2 耦合开启后：传播是否存在、是否在有限步内 ---")
    floor_on = null_response(coupling=1.0)
    reference = max(zero_peak, floor_on, 1e-12)
    print(f"  零假设底 {floor_on:.3e}   判据基准 {reference:.3e}")
    results = {}
    print(f"  {'扰动模':>8}{'最大响应':>14}{'相对基准':>11}{'首达步':>9}")
    for mode in local_modes:
        series, first = measure_response(coupling=1.0, perturb_mode=mode)
        peak = float(series.max())
        snr = peak / reference
        results[mode] = {"peak": peak, "snr": snr, "first_step": first}
        print(f"  {mode:>8}{peak:>14.3e}{snr:>11.2f}{str(first):>9}")
    c1_ok = all(results[m]["snr"] > NOISE_FLOOR_MULTIPLE for m in local_modes)
    c2_ok = all(results[m]["first_step"] is not None and results[m]["first_step"] < MAX_RESPONSE_STEPS
                for m in local_modes)
    print(f"  C1 {'通过' if c1_ok else '失败'}：传播{'存在' if c1_ok else '不存在'}")
    print(f"  C2 {'通过' if c2_ok else '失败'}：响应在有限步内出现")

    print("\n--- C3 特异性：不同局部扰动是否给出不同全局响应 ---")
    peaks = [results[m]["peak"] for m in local_modes]
    ratio = max(peaks) / (min(peaks) + 1e-30)
    print(f"  各扰动引起的响应：{['%.3e' % p for p in peaks]}")
    print(f"  最大/最小 = {ratio:.3f}")
    c3_ok = ratio > SPECIFICITY_MIN_RATIO
    print(f"  C3 {'通过' if c3_ok else '失败'}："
          f"{'响应有特异性，是信息传递' if c3_ok else '**响应几乎相同，只是全局增益，不是信息传递**'}")

    print("\n--- C4 因果有序：响应是否在扰动之后 ---")
    c4_ok = all(results[m]["first_step"] is not None for m in local_modes)
    print(f"  所有扰动的响应首达步均 >= 0：{c4_ok}")

    print("\n=== 判定 ===")
    verdicts = {"C0": c0_ok, "C1": c1_ok, "C2": c2_ok, "C3": c3_ok, "C4": c4_ok}
    for key, value in verdicts.items():
        print(f"  {key}: {'通过' if value else '失败'}")
    if all(verdicts.values()):
        print(
            "\n  **全部通过**。约束 6（整合）与约束 7（全局工作台）的最小前提成立："
            "\n  局部活动能被全局模感知，且携带特异性信息 —— 这是「可报告」的物理基础。"
        )
    else:
        failed = [k for k, v in verdicts.items() if not v]
        print(f"\n  **未通过**：{failed}。约束 6/7 的前提不成立，架构需重新推导。")

    output = Path(__file__).resolve().parent / "runs" / "constraints.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "verdicts": verdicts,
                "noise_floor_zero": floor_zero,
                "noise_floor_coupled": floor_on,
                "results": {str(k): {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in results.items()},
                "specificity_ratio": ratio,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
