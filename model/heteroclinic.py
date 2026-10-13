"""
生死线实验：共振异宿场到底有没有亚稳态？

**只回答一个问题**，不做训练、不做读出、不定判据之外的任何东西：

    轨迹在模空间里的**停留时间分布**长什么样？

判据（事先定死，不许事后挪）：
    单峰窄分布  -> 只有指数衰减，**没有亚稳态，没有链**，整个设计作废
    多峰/长尾   -> 存在亚稳态，异宿链的前提成立，值得继续

## 动力学的三个部件，全部对应书本算子

1. **色散传播**（Translation）：φ_k ← φ_k − ω_k·dt，ω_k = 2πk/L
   每个模按自身频率积累相位。取等距栅格，是为了让共振结构落在 ℤ/L 上。

2. **三波混频**（周期卷积的 Ω = ω₁+ω₂）：`k₁+k₂=k₃` 在**空间域是逐点相乘**，
   因为 k 域的卷积对应 x 域的乘积。所以就是 `ifft(ψ)²` 再变换回来。

3. **固定能量归一**：这是产生「竞争」而非「衰减」的关键。
   总能量守恒时，模与模之间是**零和**的——一个模占优必然压制别的。
   没有这一步，所有模一起衰减，永远不会有亚稳态。
   这正是 winnerless competition（Rabinovich）的做法。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def run_field(
    modes: int = 256,
    steps: int = 20000,
    dt: float = 0.05,
    coupling: float = 1.2,
    damping_low: float = 0.9995,     # 慢模的每步衰减
    damping_high: float = 0.985,     # 快模的每步衰减
    saturation: float = 1.0,
    energy: float = 1.0,
    seed: int = 0,
):
    """跑一次共振异宿场，返回逐步的模功率与场本身。"""
    rng = np.random.default_rng(seed)
    last = modes - 1
    k = np.arange(modes)

    # 色散关系：等距栅格，使共振结构落在 ℤ/L 上
    omega = 2 * math.pi * k / modes
    # 慢模记得久，快模衰减快 —— 多时间尺度是结构性的
    normalized = k / last
    damping = damping_low + (damping_high - damping_low) * normalized

    # 初始：随机相位的小扰动（对称初值会让共振结构不被激发）
    field = (rng.normal(size=modes) + 1j * rng.normal(size=modes)) * 0.1
    field /= np.linalg.norm(field)
    field *= math.sqrt(energy)

    history = np.empty((steps, modes), dtype=np.float32)

    for step in range(steps):
        # 1) 色散传播（Translation）＋ 逐模耗散
        field = field * np.exp(-1j * omega * dt) * damping
        # 2) 三波混频：空间域逐点相乘 == k₁+k₂=k₃
        spatial = np.fft.ifft(field)
        # **关键**：空间域幅值是 O(1/√L)（L=256 时约 0.06），直接送进 tanh 会落在
        # 线性区，tanh(0.06)²≈0.0036，耦合项量级 1e-4，等于没有非线性。
        # 先归一到单位 RMS 再施加饱和，最后按原尺度还原 —— 这样无论 L 多大，
        # 非线性都工作在饱和区。上一版的"对照与基线完全相同"就是这个量级错误造成的。
        rms = np.sqrt(np.mean(np.abs(spatial) ** 2))
        if rms > 1e-12:
            normalized_spatial = spatial / rms * saturation
            mixed = np.tanh(normalized_spatial) ** 2 * (rms * rms)
        else:
            mixed = np.zeros_like(spatial)
        field = field + coupling * dt * np.fft.fft(mixed)
        # 3) 固定能量：制造零和竞争，这是亚稳态的来源
        norm = np.linalg.norm(field)
        if norm > 0:
            field = field / norm * math.sqrt(energy)
        history[step] = np.abs(field) ** 2

    return history, omega, damping


def dwell_times(power: np.ndarray, smooth: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    轨迹在模空间中的停留时间。

    每步取功率最大的模作为「当前所在的想法」，连续同一想法的步数即一次停留。
    这是对异宿轨迹最朴素的读出方式 —— 若连它都测不出亚稳态，
    更精细的读出也不必试了。
    """
    # 轻度时间平滑，滤掉单步噪声（否则「停留」全是 1）
    kernel = np.ones(smooth, dtype=np.float32) / smooth
    smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, power)
    dominant = np.argmax(smoothed, axis=1)

    runs = []
    current = dominant[0]
    length = 1
    for value in dominant[1:]:
        if value == current:
            length += 1
        else:
            runs.append(length)
            current = value
            length = 1
    runs.append(length)
    return np.array(runs), dominant


def summarise(name: str, runs: np.ndarray, dominant: np.ndarray, modes: int) -> dict:
    if runs.size == 0:
        print(f"  {name}: 无停留记录")
        return {}
    distinct = int(np.unique(dominant).size)
    # 平稳性：后半段的停留分布是否与前半段相同
    half = dominant.size // 2
    front = np.unique(dominant[:half]).size
    back = np.unique(dominant[half:]).size
    result = {
        "name": name,
        "transitions": int(runs.size),
        "mean_dwell": float(runs.mean()),
        "median_dwell": float(np.median(runs)),
        "max_dwell": int(runs.max()),
        "stdev_dwell": float(runs.std()),
        # 变异系数：接近 0 = 单峰窄分布（无亚稳态）；明显大于 0 = 有结构
        "cv_dwell": float(runs.std() / runs.mean()) if runs.mean() > 0 else 0.0,
        "distinct_modes_visited": distinct,
        "fraction_of_modes": distinct / modes,
        "front_unique": int(front),
        "back_unique": int(back),
    }
    print(
        f"  {name:<22} 停留 {result['mean_dwell']:7.2f} ± {result['stdev_dwell']:6.2f} 步"
        f"  最大 {result['max_dwell']:>6}  转移 {result['transitions']:>7,}"
        f"  访问模 {distinct:>4}/{modes}  CV {result['cv_dwell']:.3f}"
    )
    return result


def main() -> None:
    print("=== 共振异宿场：停留时间分布（生死线）===\n")
    print("判据：单峰窄分布 => 只有衰减，无亚稳态，设计作废")
    print("      多峰/长尾（CV 明显>0，最大停留远大于均值）=> 存在亚稳态\n")

    rows = []
    for label, kwargs in [
        ("基线 L=256", dict(modes=256, coupling=1.2)),
        ("弱耦合", dict(modes=256, coupling=0.3)),
        ("强耦合", dict(modes=256, coupling=3.0)),
        ("无耦合（对照）", dict(modes=256, coupling=0.0)),
        ("小 L=64", dict(modes=64, coupling=1.2)),
        ("大 L=512", dict(modes=512, coupling=1.2)),
        ("慢衰减", dict(modes=256, coupling=1.2, damping_high=0.999)),
        ("快衰减", dict(modes=256, coupling=1.2, damping_high=0.95)),
    ]:
        power, _, _ = run_field(**kwargs)
        runs, dominant = dwell_times(power)
        row = summarise(label, runs, dominant, kwargs.get("modes", 256))
        if row:
            row["config"] = {k: v for k, v in kwargs.items()}
            rows.append(row)

    print("\n=== 判定 ===")
    informative = [r for r in rows if "无耦合" not in r["name"]]
    if not informative:
        print("  无有效数据")
        return
    best = max(informative, key=lambda r: r["cv_dwell"])
    print(f"  变异系数最大者：{best['name']}  CV={best['cv_dwell']:.3f}")
    print(f"  最大停留 {best['max_dwell']} 步，均值 {best['mean_dwell']:.2f} 步")
    no_coupling = next((r for r in rows if "无耦合" in r["name"]), None)
    if no_coupling:
        print(
            f"  对照（无耦合）：CV={no_coupling['cv_dwell']:.3f}，最大停留 {no_coupling['max_dwell']} 步"
        )
        gain = best["max_dwell"] / max(1, no_coupling["max_dwell"])
        print(f"  耦合使最大停留延长 {gain:.1f}×")

    if best["transitions"] < 50:
        print(
            f"\n  **判定：无效。** 只有 {best['transitions']} 次转移，"
            "\n  轨迹基本冻结。在这么少的样本上算 CV 没有意义 ——"
            "\n  这正是上一轮情感层判据犯过的错（阈值太松，在退化情形上也判通过）。"
        )
    elif best["fraction_of_modes"] < 0.05:
        print(
            f"\n  **判定：未通过。** 只访问了 {best['fraction_of_modes'] * 100:.1f}% 的模，"
            "\n  场没有在模空间里展开，谈不上轨迹，更谈不上链。"
        )
    elif best["cv_dwell"] < 0.35:
        print(
            "\n  **判定：未通过。** 停留时间接近单峰窄分布，"
            "\n  场主要在做指数衰减，亚稳态没有形成 —— 异宿链的前提不成立。"
        )
    else:
        print(
            "\n  判定：**初步通过**。转移次数、模覆盖、停留长尾三项都满足，"
            "\n  存在亚稳态迹象。下一步才值得做「转移是否由相位匹配选择」。"
        )

    output = Path(__file__).resolve().parent / "runs" / "heteroclinic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
