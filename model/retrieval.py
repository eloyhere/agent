"""
约束体系的第三块试金石：干涉 —— 内容可寻址的全息检索。

## 为什么这是约束 2

前两块验的是约束 6、7（整合与全局工作台）。它们说的是「场能连通」。
但连通不等于**有内容**。约束 2 要求的是：

    给一个残缺/带噪的线索，场能**收敛到正确的存储模式**。

这就是全息记忆的定义，而它在书本里的算子形式是明确的：

    存储（绑定）  ψ = Σᵢ pᵢ                    ← 叠加，约束 1
    匹配（解绑）  cᵢ = ⟨pᵢ, ψ⟩                 ← **趋势卷积 𝕋_mn = c_m conj(d_n)**
    重建          ψ ← f(Σᵢ cᵢ pᵢ)              ← 干涉 + 非线性

**注意 `⟨pᵢ, ψ⟩` 就是书本的趋势卷积。** 所以这不是"外挂一个记忆模块"，
而是约束 1 与约束 2 的联合实现。

## 判据（先写下来；特异性用**方向敏感**的量）

    E0  随机线索对照   给纯噪声线索时，检索正确率必须约等于随机猜测 1/M
    E1  检索存在       给带噪线索，正确率显著高于随机
    E2  **特异性**     用**检索到哪个模式**判（方向敏感），不用差异向量的长度
                       —— 上一轮 D2 就是用长度量的，看不出方向，白判了一次
    E3  容量           正确率随存储数 M 的变化
    E4  噪声鲁棒性     正确率随线索损坏程度的变化

E0 是零对照，E2 上一轮栽过，两者是重点。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RETRIEVAL_STEPS = 24
SUCCESS_THRESHOLD = 0.9        # E1：正确率下限
CHANCE_TOLERANCE = 3.0         # E0：允许超过随机猜测的倍数


def make_patterns(count: int, modes: int, seed: int = 0) -> np.ndarray:
    """存储模式。用随机复相量，单位模长 —— 这样内积就是纯粹的相位对齐度量。"""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * math.pi, (count, modes))
    return np.exp(1j * phase)


def retrieve(memory: np.ndarray, cue: np.ndarray, steps: int = RETRIEVAL_STEPS, gain: float = 2.0):
    """
    全息检索动力学。

        cᵢ = ⟨pᵢ, ψ⟩          ← 趋势卷积（书本算子）
        ψ ← f(Σᵢ cᵢ pᵢ)       ← 干涉 + 非线性

    返回 (最终场, 每步的匹配系数)。
    """
    state = cue.astype(np.complex128)
    history = []
    for _ in range(steps):
        # 匹配：这一步就是书本的趋势卷积
        coefficients = memory.conj() @ state / memory.shape[1]
        history.append(coefficients.copy())
        # 重建 + 非线性（相位饱和，使场趋向单位模长的模式）
        reconstructed = memory.T @ coefficients
        state = np.tanh(gain * np.abs(reconstructed)) * np.exp(1j * np.angle(reconstructed))
    return state, np.array(history)


def identify(memory: np.ndarray, state: np.ndarray) -> int:
    """
    **方向敏感**的判据：返回与场最对齐的模式下标。

    用 |⟨pᵢ, ψ⟩|/‖ψ‖ 归一化，只看方向不看长度 —— 上一轮 D2 就是在这里
    用差异向量的长度当特异性，看不见方向差别。
    """
    norm = np.linalg.norm(state)
    if norm <= 0:
        return -1
    overlaps = np.abs(memory.conj() @ state) / (memory.shape[1] * norm)
    return int(np.argmax(overlaps))


def trial(memory: np.ndarray, target: int, corruption: float, rng: np.random.Generator) -> bool:
    """一次检索试验。corruption=1 表示线索与噪声等能量。"""
    modes = memory.shape[1]
    noise = rng.normal(size=modes) + 1j * rng.normal(size=modes)
    noise /= np.linalg.norm(noise)
    noise *= math.sqrt(modes)
    cue = memory[target] + corruption * noise
    state, _ = retrieve(memory, cue)
    return identify(memory, state) == target


def accuracy(memory: np.ndarray, corruption: float, trials: int = 40, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    hits = sum(trial(memory, int(rng.integers(0, memory.shape[0])), corruption, rng) for _ in range(trials))
    return hits / trials


def main() -> None:
    modes = 256
    print("=== 约束体系第三块试金石：全息检索（约束 2）===\n")

    # ---------- E0：随机线索对照
    print("--- E0 零对照：给纯噪声线索，正确率应约等于随机猜测 ---")
    count = 16
    memory = make_patterns(count, modes, seed=1)
    chance = 1.0 / count
    # corruption 极大 == 线索几乎是纯噪声
    noise_accuracy = accuracy(memory, corruption=50.0, trials=60, seed=7)
    print(f"  存储 {count} 个模式，随机猜测基线 {chance:.4f}")
    print(f"  纯噪声线索的正确率 {noise_accuracy:.4f}")
    e0_ok = noise_accuracy <= CHANCE_TOLERANCE * chance
    print(f"  E0 {'通过' if e0_ok else '失败'}："
          f"{'接近随机猜测（符合预期）' if e0_ok else '**显著高于随机，说明读出环节有偏**'}")

    # ---------- E1：检索存在
    print("\n--- E1 检索存在：给带噪线索，正确率应显著高于随机 ---")
    print(f"  {'存储数 M':>10}{'正确率':>10}{'随机基线':>10}")
    capacity = []
    for count in (4, 8, 16, 32, 64):
        memory = make_patterns(count, modes, seed=1)
        acc = accuracy(memory, corruption=0.5, trials=40, seed=11)
        capacity.append({"count": count, "accuracy": acc, "chance": 1.0 / count})
        print(f"  {count:>10}{acc:>10.4f}{1.0 / count:>10.4f}")
    e1_ok = all(row["accuracy"] > 5 * row["chance"] for row in capacity if row["count"] <= 32)
    print(f"  E1 {'通过' if e1_ok else '失败'}："
          f"{'检索确实发生' if e1_ok else '**正确率未显著高于随机**'}")

    # ---------- E2：特异性（方向敏感）
    print("\n--- E2 特异性：用「检索到哪个模式」判，不用差异向量的长度 ---")
    count = 16
    memory = make_patterns(count, modes, seed=1)
    rng = np.random.default_rng(23)
    retrieved = []
    for target in range(count):
        state, _ = retrieve(memory, memory[target] + 0.3 * (rng.normal(size=modes) + 1j * rng.normal(size=modes)) / math.sqrt(modes))
        retrieved.append(identify(memory, state))
    distinct = len(set(retrieved))
    correct = sum(1 for i, r in enumerate(retrieved) if r == i)
    print(f"  16 个线索 -> 检索到 {distinct} 个**不同**的模式下标")
    print(f"  其中命中原目标的 {correct}/16")
    e2_ok = distinct == count
    print(f"  E2 {'通过' if e2_ok else '失败'}："
          f"{'不同线索确实检索到不同模式（方向敏感）' if e2_ok else '**出现模式混淆**'}")

    # ---------- E3/E4：容量与噪声
    print("\n--- E4 噪声鲁棒性（M=16）---")
    memory = make_patterns(16, modes, seed=1)
    print(f"  {'损坏程度':>10}{'正确率':>10}")
    noise_rows = []
    for corruption in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        acc = accuracy(memory, corruption=corruption, trials=40, seed=31)
        noise_rows.append({"corruption": corruption, "accuracy": acc})
        print(f"  {corruption:>10.2f}{acc:>10.4f}")
    e4_ok = noise_rows[2]["accuracy"] > 0.5
    print(f"  E4 {'通过' if e4_ok else '失败'}：中等损坏下仍能检索")

    print("\n=== 判定 ===")
    verdicts = {"E0": e0_ok, "E1": e1_ok, "E2": e2_ok, "E4": e4_ok}
    for key, value in verdicts.items():
        print(f"  {key}: {'通过' if value else '失败'}")
    if all(verdicts.values()):
        print(
            "\n  **全部通过**。约束 2 成立：场能以内容寻址的方式检索存储模式，"
            "\n  且判据是方向敏感的 —— 不同线索检索到不同模式，不是笼统的响应。"
        )
    else:
        failed = [k for k, v in verdicts.items() if not v]
        print(f"\n  **未通过**：{failed}。")

    output = Path(__file__).resolve().parent / "runs" / "retrieval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "verdicts": verdicts,
                "noise_accuracy": noise_accuracy,
                "chance": chance,
                "capacity": capacity,
                "noise_robustness": noise_rows,
                "distinct_retrieved": distinct,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
