"""
切分实验：连续经验能不能切成可复用、可复现的时序模式？

这是贝叶斯框架的地基。如果切不出稳定的模式，"先验"就无从谈起，
后面的似然、后验、MAP 路径全部悬空。

## 方法：匹配峰即边界

时序匹配在离散情形下最朴素的实现就是**最长匹配**（LZ77 那一路）：
在位置 i，找它与**更早某个位置**的最长公共前缀长度 `match[i]`。

    match[i] 大  ->  这里开始的内容以前出现过 == 匹配峰
    match[i-1] 小 ->  之前没有这段内容 == 峰的起点

于是**在"由低转高"处切分**：切出来的每一段都是一个被复用的模式。

## 判据（事先定死）

    可复用性：切出的片段中，重复出现的比例
    一致性  ：同一段文本在不同位置出现时，切分边界是否落在**相同的相对位置**

一致性是生死线。若同一句话两次出现却切得不一样，那"模式"就不是模式，
只是随上下文漂移的产物 —— 整个贝叶斯先验没有稳定指称。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def longest_matches(data: bytes, min_seed: int = 4, max_extend: int = 4096) -> list[int]:
    """
    对每个位置，返回与更早位置的最长公共前缀长度。

    用 min_seed 长的种子做索引，再向前延伸。这是标准 LZ77 匹配。
    """
    n = len(data)
    index: dict[bytes, list[int]] = defaultdict(list)
    matches = [0] * n
    for i in range(n):
        if i + min_seed <= n:
            seed = data[i : i + min_seed]
            candidates = index[seed]
            best = 0
            for j in candidates:
                # 延伸
                length = min_seed
                limit = min(n - i, n - j, max_extend)
                while length < limit and data[i + length] == data[j + length]:
                    length += 1
                if length > best:
                    best = length
                    if best >= limit:
                        break
            matches[i] = best
            index[seed].append(i)
    return matches


def segment(data: bytes, matches: list[int], min_match: int, min_length: int) -> list[tuple[int, int]]:
    """在匹配由低转高处切分。"""
    boundaries = [0]
    for i in range(1, len(matches)):
        if matches[i] >= min_match and matches[i - 1] < min_match:
            if i - boundaries[-1] >= min_length:
                boundaries.append(i)
    boundaries.append(len(matches))
    return [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]


def evaluate(data: bytes, segments: list[tuple[int, int]]) -> dict:
    pieces = [data[a:b] for a, b in segments]
    counts: dict[bytes, int] = defaultdict(int)
    for piece in pieces:
        counts[piece] += 1
    distinct = len(counts)
    reused = sum(1 for c in counts.values() if c > 1)
    total = len(pieces)
    lengths = [b - a for a, b in segments]
    return {
        "total_segments": total,
        "distinct_segments": distinct,
        "reused_segments": reused,
        "reuse_fraction": reused / total if total else 0.0,
        "mean_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_length": max(lengths) if lengths else 0,
        "coverage": sum(lengths) / len(data) if len(data) else 0.0,
    }


def consistency(data: bytes, segments: list[tuple[int, int]], probe_length: int = 24, min_occurrences: int = 3) -> dict:
    """
    **生死线**：同一段文本在不同位置出现，切分边界是否落在相同的相对位置。

    做法：取若干在语料里多次出现的 probe 串，对每个出现位置，记录
    「在该出现内部，相对 probe 起点的切分边界偏移量」。若一致，
    则同一个模式被重复切成了同样的形状。
    """
    boundaries = sorted(b for a, b in segments)
    occurrence_map: dict[bytes, list[int]] = defaultdict(list)
    n = len(data)
    for i in range(0, n - probe_length):
        occurrence_map[data[i : i + probe_length]].append(i)

    probes = [(s, pos) for s, pos in occurrence_map.items() if len(pos) >= min_occurrences]
    # 取出现次数最多的若个 probe
    probes.sort(key=lambda item: -len(item[1]))
    probes = probes[:20]

    consistent = 0
    inconsistent = 0
    details = []
    import bisect

    for probe, positions in probes:
        patterns = []
        for start in positions:
            left = bisect.bisect_left(boundaries, start)
            right = bisect.bisect_right(boundaries, start + probe_length)
            # probe 内部的切分边界，转成相对偏移
            relative = tuple(b - start for b in boundaries[left:right] if start < b < start + probe_length)
            patterns.append(relative)
        if len(set(patterns)) == 1:
            consistent += 1
        else:
            inconsistent += 1
        details.append(
            {
                "probe": probe.decode("utf-8", errors="replace"),
                "occurrences": len(positions),
                "distinct_patterns": len(set(patterns)),
                "sample": [list(p) for p in patterns[:3]],
            }
        )

    return {
        "probes_checked": len(probes),
        "consistent": consistent,
        "inconsistent": inconsistent,
        "consistency_rate": consistent / len(probes) if probes else 0.0,
        "details": details[:6],
    }


def main() -> None:
    corpus_root = Path(__file__).resolve().parents[2] / "periodicity"
    chunks: list[bytes] = []
    for path in sorted(corpus_root.rglob("*.tex"))[:40]:
        try:
            chunks.append(path.read_bytes())
        except OSError:
            continue
    data = b"\n".join(chunks)
    # 截一段可控长度
    limit = 200_000
    data = data[:limit]
    print(f"语料 {len(data):,} 字节\n")

    print("=== 先看：最长匹配随位置的变化（前 300 字节）===")
    matches = longest_matches(data)
    preview = "".join("#" if m >= 8 else ("+" if m >= 4 else ".") for m in matches[:300])
    print("  " + preview[:150])
    print("  " + preview[150:300])
    print("  图例：# 匹配≥8   + 匹配≥4   . 匹配<4\n")

    rows = []
    for min_match in (6, 8, 12, 16):
        segments = segment(data, matches, min_match=min_match, min_length=8)
        stats = evaluate(data, segments)
        stats["min_match"] = min_match
        rows.append(stats)
        print(
            f"  min_match={min_match:>3}  片段 {stats['total_segments']:>6,}"
            f"  去重后 {stats['distinct_segments']:>6,}"
            f"  可复用 {stats['reuse_fraction'] * 100:>5.1f}%"
            f"  平均长 {stats['mean_length']:>6.1f}"
            f"  覆盖 {stats['coverage'] * 100:>5.1f}%"
        )

    best = max(rows, key=lambda r: r["reuse_fraction"])
    print(f"\n  可复用性最高：min_match={best['min_match']}  ({best['reuse_fraction'] * 100:.1f}%)")

    print("\n=== 生死线：切分一致性 ===")
    segments = segment(data, matches, min_match=best["min_match"], min_length=8)
    result = consistency(data, segments)
    print(f"  检查 {result['probes_checked']} 个多次出现的探针串")
    print(f"  切分一致 {result['consistent']}  不一致 {result['inconsistent']}")
    print(f"  一致率 {result['consistency_rate'] * 100:.1f}%\n")
    for detail in result["details"]:
        print(
            f"    {detail['probe']!r:<30} 出现 {detail['occurrences']:>4} 次"
            f"  切法 {detail['distinct_patterns']} 种  例 {detail['sample'][:2]}"
        )

    print("\n=== 判定 ===")
    if result["probes_checked"] == 0:
        print("  **无效**：找不到足够多次出现的探针串，无法评估一致性。")
    elif result["consistency_rate"] >= 0.8:
        print(
            f"  判定：**通过**。一致率 {result['consistency_rate'] * 100:.0f}%，"
            "\n  同一段文本在不同位置被切成同样的形状 —— 时序模式有稳定指称，"
            "\n  贝叶斯先验有地基。"
        )
    elif result["consistency_rate"] >= 0.5:
        print(
            f"  判定：**部分通过**。一致率 {result['consistency_rate'] * 100:.0f}%，"
            "\n  模式部分稳定，但受上下文影响明显。可作为弱先验使用。"
        )
    else:
        print(
            f"  判定：**未通过**。一致率仅 {result['consistency_rate'] * 100:.0f}%，"
            "\n  同一段文本在不同位置切法不同 —— 切分随上下文漂移，"
            "\n  切出来的不是「模式」而是上下文的产物。贝叶斯先验没有稳定指称。"
        )

    output = Path(__file__).resolve().parent / "runs" / "segmentation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"stats": rows, "consistency": result}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
