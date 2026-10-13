"""
修正 E1 的判据，并**检验「串扰」这个诊断本身是对是错**。

## 上一轮留下的两个问题

1. **E1 判据不可达**：我写了 `正确率 > 5 × 随机基线`。M=4 时基线 0.25，要求
   正确率 > 1.25，而正确率上限是 1.0 —— **这个判据在数学上不可能通过**。
   换成绝对阈值 + 统计显著性，不再用「基线的倍数」这种在基线大时失效的形式。

2. **E4 的正确率对噪声完全持平（0.725 前四档不变）**，说明失败不是噪声造成的。
   我当时的诊断是「模式间串扰」——**但那是推测，没有验证**。

## 本轮的判据（先写下来）

    F1  判据可达性   新判据在正确率上限（1.0）处必须**可满足**
    F2  串扰诊断     **每个模式的检索正确率应随它与最近邻的互相关单调下降**
                     —— 这是对我上一轮那个推测的直接检验。若不下降，诊断作废。
    F3  噪声效应     在**低互相关**的模式上，正确率应对噪声敏感
                     —— 若连低互相关模式也对噪声不敏感，说明还有别的原因没找到
    F4  样本量       每个模式足够多次试验，使标准误小于组间差异

F2 是核心。它是对我自己上一轮结论的证伪尝试，不是对它的确认。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RETRIEVAL_STEPS = 24
TRIALS_PER_TARGET = 60      # F4：比上一轮的 40 次总量大幅提高


def make_patterns(count: int, modes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(0, 2 * math.pi, (count, modes)))


def correlation_matrix(memory: np.ndarray) -> np.ndarray:
    """模式间归一化互相关 |⟨pᵢ,pⱼ⟩|/L，对角线置 0。"""
    modes = memory.shape[1]
    gram = np.abs(memory.conj() @ memory.T) / modes
    np.fill_diagonal(gram, 0.0)
    return gram


def retrieve(memory: np.ndarray, cue: np.ndarray, steps: int = RETRIEVAL_STEPS, gain: float = 2.0):
    state = cue.astype(np.complex128)
    for _ in range(steps):
        coefficients = memory.conj() @ state / memory.shape[1]
        reconstructed = memory.T @ coefficients
        state = np.tanh(gain * np.abs(reconstructed)) * np.exp(1j * np.angle(reconstructed))
    return state


def identify(memory: np.ndarray, state: np.ndarray) -> int:
    norm = np.linalg.norm(state)
    if norm <= 0:
        return -1
    return int(np.argmax(np.abs(memory.conj() @ state) / (memory.shape[1] * norm)))


def per_target_accuracy(memory: np.ndarray, corruption: float, trials: int, seed: int) -> np.ndarray:
    """对**每一个**目标模式分别统计正确率。"""
    count, modes = memory.shape
    rng = np.random.default_rng(seed)
    hits = np.zeros(count)
    for target in range(count):
        for _ in range(trials):
            noise = rng.normal(size=modes) + 1j * rng.normal(size=modes)
            noise /= np.linalg.norm(noise)
            noise *= math.sqrt(modes)
            cue = memory[target] + corruption * noise
            if identify(memory, retrieve(memory, cue)) == target:
                hits[target] += 1
    return hits / trials


def main() -> None:
    modes = 256
    count = 16
    memory = make_patterns(count, modes, seed=1)
    gram = correlation_matrix(memory)
    max_corr = gram.max(axis=1)

    print("=== 检验「串扰」诊断：正确率是否随互相关下降 ===\n")
    print(f"  模式数 {count}，维数 {modes}，每目标 {TRIALS_PER_TARGET} 次试验")
    print(f"  互相关：均值 {gram[gram > 0].mean():.4f}  最大 {max_corr.max():.4f}"
          f"  最小 {max_corr.min():.4f}")

    # ---------- F2：核心检验
    print("\n--- F2 核心检验：每个模式的正确率 vs 它与最近邻的互相关 ---")
    clean = per_target_accuracy(memory, corruption=0.0, trials=TRIALS_PER_TARGET, seed=101)
    order = np.argsort(max_corr)
    print(f"  {'模式':>5}{'最近邻互相关':>14}{'干净线索正确率':>16}")
    for index in order:
        print(f"  {index:>5}{max_corr[index]:>14.4f}{clean[index]:>16.4f}")

    # 相关性与分组对比
    if clean.std() > 0:
        correlation = float(np.corrcoef(max_corr, clean)[0, 1])
    else:
        correlation = float("nan")
    half = count // 2
    low_group = clean[order[:half]]
    high_group = clean[order[half:]]
    print(f"\n  低互相关组（一半）正确率均值 {low_group.mean():.4f}")
    print(f"  高互相关组（一半）正确率均值 {high_group.mean():.4f}")
    print(f"  corr(最近邻互相关, 正确率) = {correlation:+.4f}")
    f2_ok = correlation < -0.3 and low_group.mean() > high_group.mean()
    print(f"  F2 {'通过' if f2_ok else '失败'}："
          f"{'串扰诊断成立 —— 正确率确实随互相关下降' if f2_ok else '**正确率未随互相关下降，串扰诊断作废，需另找原因**'}")

    # ---------- F3：低互相关模式是否对噪声敏感
    print("\n--- F3 低互相关模式对噪声是否敏感 ---")
    print(f"  {'损坏程度':>10}{'低互相关组':>14}{'高互相关组':>14}")
    noise_rows = []
    for corruption in (0.0, 0.5, 1.0, 2.0, 4.0):
        per_target = per_target_accuracy(memory, corruption=corruption,
                                         trials=TRIALS_PER_TARGET, seed=202)
        low = per_target[order[:half]].mean()
        high = per_target[order[half:]].mean()
        noise_rows.append({"corruption": corruption, "low": float(low), "high": float(high)})
        print(f"  {corruption:>10.2f}{low:>14.4f}{high:>14.4f}")
    low_drop = noise_rows[0]["low"] - noise_rows[-1]["low"]
    f3_ok = low_drop > 0.05
    print(f"  低互相关组从损坏 0 到 4 下降 {low_drop:.4f}")
    print(f"  F3 {'通过' if f3_ok else '失败'}："
          f"{'对噪声敏感，说明串扰之外还有噪声效应' if f3_ok else '**即使低互相关也不受噪声影响，另有原因**'}")

    # ---------- F1：新判据可达性与统计显著性
    print("\n--- F1 修正后的 E1 判据 ---")
    print("  旧判据 `正确率 > 5 × 随机基线`：M=4 时要求 > 1.25，**不可达**")
    print("  新判据 `正确率 > 0.5 且显著高于随机（二项检验，p < 1e-6）`")
    print(f"  {'M':>6}{'正确率':>10}{'随机':>9}{'是否可达':>10}{'显著':>8}")
    capacity = []
    for m in (4, 8, 16, 32, 64):
        mem = make_patterns(m, modes, seed=1)
        rng = np.random.default_rng(303)
        trials = 400
        hits = 0
        for _ in range(trials):
            target = int(rng.integers(0, m))
            noise = rng.normal(size=modes) + 1j * rng.normal(size=modes)
            noise /= np.linalg.norm(noise)
            noise *= math.sqrt(modes)
            if identify(mem, retrieve(mem, mem[target] + 0.5 * noise)) == target:
                hits += 1
        accuracy = hits / trials
        chance = 1.0 / m
        # 二项检验的正态近似
        standard_error = math.sqrt(chance * (1 - chance) / trials)
        z = (accuracy - chance) / standard_error if standard_error > 0 else 0.0
        reachable = 0.5 <= 1.0          # 判据阈值必须 <= 正确率上限
        capacity.append({"m": m, "accuracy": accuracy, "chance": chance, "z": float(z)})
        print(f"  {m:>6}{accuracy:>10.4f}{chance:>9.4f}{str(reachable):>10}{z:>8.1f}")
    f1_ok = all(row["accuracy"] > 0.5 and row["z"] > 5 for row in capacity)
    print(f"  F1 {'通过' if f1_ok else '失败'}：新判据可达且被满足")

    print("\n=== 判定 ===")
    verdicts = {"F1 判据可达": f1_ok, "F2 串扰诊断": f2_ok, "F3 噪声效应": f3_ok}
    for key, value in verdicts.items():
        print(f"  {key}: {'通过' if value else '失败'}")
    if f2_ok:
        print("\n  上一轮「容量受串扰限制」的诊断**得到证实**。")
    else:
        print("\n  上一轮那个诊断**不成立** —— 我当时的推测是错的，需另找原因。")

    output = Path(__file__).resolve().parent / "runs" / "crosstalk.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "verdicts": verdicts,
                "max_correlation": max_corr.tolist(),
                "clean_accuracy": clean.tolist(),
                "correlation": correlation,
                "noise_rows": noise_rows,
                "capacity": capacity,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
