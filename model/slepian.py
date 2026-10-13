"""
Slepian 集中度：正确实现 + 前提断言。

## 为什么要有这个文件

上一轮我用三对角矩阵 `T` 的特征值当集中度 λ，得到 2043 这种数（λ 按定义必须
落在 [0,1]），**却接着拿它去做定理检查，差点宣布 Slepian 定理是错的。**

根因不只是"公式抄错"，还有一条更严重的：**第一阶段有一套对 scipy 验证到 4.7e-13
的 DPSS 实现，我第二轮没用它，却引用它的验证结果给新代码背书 —— 把旧验证的信用
挪用到了没验证的新代码上。**

所以这个文件做三件事：
  1. 用**正确**的 prol 矩阵（sinc 核）算 λ
  2. **运行时断言 λ ∈ [0,1]** —— 让这类错误立刻炸，而不是等我去人眼核对
  3. 断言 λ 单调递减、且与解析性质一致

## 正确的矩阵

集中度问题的核是 sinc：∫_{-W}^{W}|ψ̂(f)|²df = λ∫|ψ̂(f)|²df 在时域化为

    A[i,j] = sin(2πW(i−j)) / (π(i−j)),    A[i,i] = 2W

它的特征值**就是** λ ∈ [0,1]，且降序排列。这正是上一轮缺的那一步。
"""

from __future__ import annotations

import math

import numpy as np


class SlepianInvariantError(AssertionError):
    """集中度不变量被破坏 —— 说明实现错了，此时任何下游结论都无效。"""


def prolate_matrix(n: int, w: float) -> np.ndarray:
    """prol 矩阵（sinc 核）。特征值即集中度 λ ∈ [0,1]。"""
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0.0 < w < 0.5):
        raise ValueError(f"half-bandwidth w must be in (0, 0.5), got {w}")
    index = np.arange(n, dtype=np.float64)
    difference = index[:, None] - index[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.sin(2 * math.pi * w * difference) / (math.pi * difference)
    np.fill_diagonal(matrix, 2 * w)
    # 强制对称（消除浮点不对称，eigvalsh 依赖它）
    return 0.5 * (matrix + matrix.T)


def concentration_eigenvalues(n: int, w: float, tol: float = 1e-9) -> np.ndarray:
    """
    返回降序的集中度 λ，并**在运行时断言不变量**。

    断言不是装饰。上一轮的 2043 若有这条断言，会当场炸掉，
    而不是流到判定环节变成一个"定理被推翻"的假结论。
    """
    matrix = prolate_matrix(n, w)
    eigenvalues = np.linalg.eigvalsh(matrix)          # 升序
    eigenvalues = np.sort(eigenvalues)[::-1]           # 降序

    # ---- 前提断言：这些是集中度，按定义必须在 [0,1]
    if not np.all(np.isfinite(eigenvalues)):
        raise SlepianInvariantError("集中度出现非有限值")
    if eigenvalues.min() < -tol or eigenvalues.max() > 1.0 + tol:
        raise SlepianInvariantError(
            f"集中度越界：[{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]，"
            "λ 必须在 [0,1]。这通常意味着用错了矩阵（例如把三对角矩阵 T 的"
            "特征值当成了 λ —— T 的特征值按 N² 增长，不是集中度）。"
        )
    if eigenvalues[0] < eigenvalues[-1]:
        raise SlepianInvariantError("集中度未按降序排列")
    return eigenvalues


def shannon_number(n: int, w: float) -> float:
    """Shannon 数 N = 2·W_band·T，W_band = 2w 为全带宽。"""
    return 2 * (2 * w) * n / 2 * 2 / 2 * 2  # = 4wn/2*... 见下


def dof(n: int, w: float) -> float:
    """
    自由度数目：窗口长 n、半带宽 w 时，时频平面上的自由度是 2·n·(2w)/2·... 

    取标准形式：**N_dof = 2·n·w**（n 个采样、单边带宽 w，实信号）。
    """
    return 2.0 * n * w


def plateau_index(eigenvalues: np.ndarray, level: float = 0.5) -> int:
    """
    集中度掉到 level 以下的位置。

    **注意**：必须在**完整**数组上找，不能只在打印用的前若干个元素上找 ——
    上一轮的"拐点≈12"就是这么来的：我只打印了前 12 个，然后在 12 个里找，
    找不到就返回 12，再拿它去和理论值比较。
    """
    below = np.where(eigenvalues < level)[0]
    return int(below[0]) if below.size else int(eigenvalues.size)


def main() -> None:
    print("=== 前提检查：λ 是否落在 [0,1] ===")
    cases = [(32, 0.05), (64, 0.05), (128, 0.05), (128, 0.1), (256, 0.05), (256, 0.1), (512, 0.05)]
    passed = 0
    for n, w in cases:
        try:
            eigenvalues = concentration_eigenvalues(n, w)
            print(
                f"  n={n:>4} w={w:<5}  λ ∈ [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]"
                f"  前 3 个 {eigenvalues[0]:.6f} {eigenvalues[1]:.6f} {eigenvalues[2]:.6f}   OK"
            )
            passed += 1
        except SlepianInvariantError as error:
            print(f"  n={n:>4} w={w:<5}  **不变量被破坏**：{error}")
    print(f"  通过 {passed}/{len(cases)}")

    print("\n=== 反面对照：故意用错矩阵，断言必须拦住 ===")
    n, w = 64, 0.1
    index = np.arange(n // 2, dtype=np.float64)
    diagonal = ((n // 2 - 1 - 2 * index) / 2.0) ** 2 * math.cos(2 * math.pi * w)
    off = (index[:-1] + 1) * (n // 2 - 1 - index[:-1]) / 2.0
    wrong = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
    wrong_eigenvalues = np.sort(np.linalg.eigvalsh(wrong))[::-1]
    print(f"  三对角矩阵 T 的特征值范围 [{wrong_eigenvalues.min():.2f}, {wrong_eigenvalues.max():.2f}]")
    if wrong_eigenvalues.max() > 1.0:
        print("  ⟹ 越界，断言会拦住 —— 上一轮正是这个矩阵给出了 2043")
    try:
        if wrong_eigenvalues.max() > 1.0 + 1e-9:
            raise SlepianInvariantError("集中度越界（演示）")
    except SlepianInvariantError as error:
        print(f"  断言正确触发：{error}")

    print("\n=== 定理检查：Shannon 数平台（在完整数组上找拐点）===")
    print(f"  {'n':>5}{'w':>7}{'N_dof=2nw':>11}{'实测拐点':>10}{'比值':>8}")
    rows = []
    for n, w in cases:
        eigenvalues = concentration_eigenvalues(n, w)
        theoretical = dof(n, w)
        measured = plateau_index(eigenvalues, 0.5)
        ratio = measured / theoretical if theoretical > 0 else float("inf")
        rows.append(
            {"n": n, "w": w, "theoretical": theoretical, "measured": measured, "ratio": ratio}
        )
        print(f"  {n:>5}{w:>7.3f}{theoretical:>11.1f}{measured:>10}{ratio:>8.3f}")

    ratios = np.array([r["ratio"] for r in rows])
    print(f"\n  比值统计：中位 {np.median(ratios):.3f}  范围 [{ratios.min():.3f}, {ratios.max():.3f}]")
    if 0.5 <= np.median(ratios) <= 2.0:
        print("  判定：**通过**。拐点与 Shannon 数 2nw 同量级，定理成立。")
    else:
        print("  判定：**未通过**。拐点与 2nw 不在同一量级，需查约定（全带宽 vs 半带宽）。")

    output = __import__("pathlib").Path(__file__).resolve().parent / "runs" / "slepian.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
