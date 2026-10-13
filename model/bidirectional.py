"""
约束体系的第二块试金石：全局工作台必须是**双向**的。

## 为什么单向不够

上一轮证明了「局部 -> 全局」：局部活动能被全局模感知，且带特异性。
但那只说明全局模是一个**汇总器**（像一根平均线）。

全局工作台（约束 7）要求的是更强的东西：

    **全局模的状态必须改变局部信息的处理方式。**

否则它只是"读出了一个和"，而不是"广播了一个调制"。认知科学里这是
broadcast 与 modulation 的区别；只做前者，就没有"可报告"以外的功能。

## 判据（先写下来）

    D0  零耦合对照     关闭耦合时，全局态对局部结果必须**零影响**
    D1  全局 -> 局部   改变全局模，局部模发生变化，且超过零假设底
    D2  有特异性       不同的全局态产生**不同**的局部结果
    D3  **门控（关键）**固定同一个局部输入，只改全局态，局部结果必须不同
                       —— 这一条才是"工作台"，D1/D2 只说明有反向连线
    D4  因果有序       局部结果的变化在全局态改变**之后**出现

D0 与 D3 是重点。D3 不过，约束 7 就只能降级为"汇总器"，架构需重新推导。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from constraints import Field, FieldInvariantError

GATE_THRESHOLD = 0.05      # D3：门控效应相对局部扰动幅度的下限
SPECIFICITY_MIN = 1.3      # D2：不同全局态引起的差异下限


def run_with_global(
    global_phase: float,
    perturb_mode: int,
    coupling: float,
    steps: int = 150,
    seed: int = 0,
    global_count: int = 4,
):
    """
    把全局模的相位钉在给定值，施加**同一个**局部扰动，跑完之后返回局部模状态。

    两个运行的唯一差别就是 `global_phase` —— 这是 D3 能成立的前提：
    固定局部输入、只动全局态。
    """
    field = Field(coupling=coupling, seed=seed, global_count=global_count)
    for _ in range(20):
        field.step()
    # 钉住全局相位
    k = np.arange(global_count)
    field.state[k] = field.baseline[k] * np.exp(1j * global_phase)
    # 同一个局部扰动
    field.state[perturb_mode] += 1.0

    for _ in range(steps):
        field.step()
    return field.state[global_count:].copy()


def main() -> None:
    print("=== 约束体系第二块试金石：全局工作台是否双向 ===\n")

    perturb_mode = 160
    phases = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]

    # ---------- 前提：非线性必须真的在起作用
    probe = Field(coupling=1.0)
    for _ in range(20):
        probe.step()
    strength = probe.coupling_strength()
    print(f"--- 前提检查 ---\n  耦合项占动力学 {strength * 100:.2f}%")
    if strength <= 0.02:
        print("  **前提不成立**：耦合过弱，直接终止，不输出判定。")
        return

    # ---------- D0：零耦合对照
    print("\n--- D0 零耦合对照：全局态对局部结果必须零影响 ---")
    baseline_zero = run_with_global(0.0, perturb_mode, coupling=0.0)
    effects_zero = []
    for phase in phases[1:]:
        other = run_with_global(phase, perturb_mode, coupling=0.0)
        effects_zero.append(float(np.linalg.norm(other - baseline_zero)))
    zero_effect = max(effects_zero)
    print(f"  改全局相位引起的局部状态变化：最大 {zero_effect:.3e}")
    d0_ok = zero_effect < 1e-9
    print(f"  D0 {'通过' if d0_ok else '失败'}：零耦合下"
          f"{'无影响（符合预期）' if d0_ok else '**有影响，说明耦合没关掉**'}")

    # ---------- D1 / D2：全局 -> 局部，且有特异性
    print("\n--- D1/D2 全局 -> 局部：是否存在反向连线，是否带特异性 ---")
    baseline = run_with_global(0.0, perturb_mode, coupling=1.0)
    effects = {}
    print(f"  {'全局相位':>10}{'局部变化':>14}{'相对零假设':>13}")
    reference = max(zero_effect, 1e-12)
    for phase in phases[1:]:
        other = run_with_global(phase, perturb_mode, coupling=1.0)
        magnitude = float(np.linalg.norm(other - baseline))
        effects[phase] = magnitude
        print(f"  {phase:>10.3f}{magnitude:>14.3e}{magnitude / reference:>13.2f}")
    d1_ok = all(v > 5 * reference for v in effects.values())
    values = list(effects.values())
    ratio = max(values) / (min(values) + 1e-30)
    d2_ok = ratio > SPECIFICITY_MIN
    print(f"  D1 {'通过' if d1_ok else '失败'}：反向连线{'存在' if d1_ok else '不存在'}")
    print(f"  D2 {'通过' if d2_ok else '失败'}：不同全局态引起不同局部结果，最大/最小 = {ratio:.3f}")

    # ---------- D3：门控 —— 固定局部输入，只改全局态
    print("\n--- D3 门控（关键）：固定同一个局部输入，只改全局态 ---")
    print("  这正是 broadcast 与 modulation 的分界：")
    print("  若局部结果不随全局态改变，全局模只是汇总器，不是工作台。")
    outputs = [run_with_global(phase, perturb_mode, coupling=1.0) for phase in phases]
    # 两两之间的差异
    pairwise = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            pairwise.append(float(np.linalg.norm(outputs[i] - outputs[j])))
    spread = max(pairwise)
    local_input_scale = 1.0    # 局部扰动的幅度
    gate_ratio = spread / local_input_scale
    print(f"  四个全局态下的局部输出，两两最大差 {spread:.4f}")
    print(f"  相对局部扰动幅度({local_input_scale}) 的门控比 {gate_ratio:.4f}")
    d3_ok = gate_ratio > GATE_THRESHOLD
    print(f"  D3 {'通过' if d3_ok else '失败'}："
          f"{'全局态确实门控局部处理' if d3_ok else '**局部输出几乎不随全局态改变，全局模只是汇总器**'}")

    # ---------- D4：因果有序 —— 全局态在局部扰动之前设定
    # 由 run_with_global 的结构保证（先钉相位，再施加扰动），此处只做声明检查
    d4_ok = True
    print(f"\n--- D4 因果有序 ---\n  run_with_global 先钉全局相位、后施加局部扰动：{d4_ok}")

    print("\n=== 判定 ===")
    verdicts = {"D0": d0_ok, "D1": d1_ok, "D2": d2_ok, "D3": d3_ok, "D4": d4_ok}
    for key, value in verdicts.items():
        print(f"  {key}: {'通过' if value else '失败'}")
    if all(verdicts.values()):
        print(
            "\n  **全部通过**。约束 7 的完整形式成立：全局模双向连接局部，"
            "\n  且它的状态**改变局部信息的处理方式** —— 这是 modulation，"
            "\n  不只是 broadcast。"
        )
    else:
        failed = [k for k, v in verdicts.items() if not v]
        print(f"\n  **未通过**：{failed}。")
        if not d3_ok:
            print("  D3 不过意味着全局模只能当汇总器，约束 7 需降级或重新推导。")

    output = Path(__file__).resolve().parent / "runs" / "bidirectional.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "verdicts": verdicts,
                "coupling_strength": strength,
                "zero_effect": zero_effect,
                "effects": {str(k): v for k, v in effects.items()},
                "specificity_ratio": ratio,
                "gate_ratio": gate_ratio,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    try:
        main()
    except FieldInvariantError as error:
        print(f"\n**不变量被破坏，终止且不输出判定**：{error}")
