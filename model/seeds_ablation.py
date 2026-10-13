"""
多种子复现 + 场模型消融。

上一轮拿到「同等算力下场模型严格更优」（3.05 对 5.88 bpb），但那是**单一种子**。
单次结果不能支撑"稳定更优"——训练有随机性，可能是运气。这里跑多个种子，
报均值与标准差，看两个分布是否真的分得开。

再做一轮消融：模态数 K 与层数如何影响质量。3.05 bpb 显然不是这个架构的上限，
消融是找出往哪加最划算。
"""

from __future__ import annotations

import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epicycle_lm import Config as EpicycleConfig, EpicycleLM
from wavefield import FieldConfig, WaveFieldLM

BATCH, LENGTH, VOCAB = 8, 256, 258
BASELINE = math.log(VOCAB) / math.log(2)


def gather_corpus(root: Path) -> bytes:
    chunks: list[bytes] = []
    for path in sorted(root.rglob("*.tex")):
        try:
            chunks.append(path.read_bytes())
        except OSError:
            continue
        chunks.append(b"\n")
    return b"".join(chunks)


def batch_of(data, device):
    starts = torch.randint(0, data.numel() - LENGTH - 1, (BATCH,))
    inputs = torch.stack([data[s : s + LENGTH] for s in starts]).to(device)
    targets = torch.stack([data[s + 1 : s + LENGTH + 1] for s in starts]).to(device)
    return inputs, targets


@torch.no_grad()
def evaluate(model, data, device, rounds=8) -> float:
    model.eval()
    total = 0.0
    for _ in range(rounds):
        inputs, targets = batch_of(data, device)
        logits = model(inputs)
        total += F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1)).item() / math.log(2)
    model.train()
    return total / rounds


def run(factory, train_data, val_data, device, seed, budget, lr=3e-3):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = factory().to(device)
    parameters = model.parameter_count()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    steps = 0
    while time.perf_counter() - started < budget:
        inputs, targets = batch_of(train_data, device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    bits = evaluate(model, val_data, device)
    del model, optimizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "parameters": parameters,
        "steps": steps,
        "seconds": elapsed,
        "peak_gb": peak,
        "bits_per_byte": bits,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}  随机基线 {BASELINE:.4f} bits/byte")
    raw = gather_corpus(Path(__file__).resolve().parents[2] / "periodicity")
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    split = int(data.numel() * 0.98)
    train_data, val_data = data[:split], data[split:]
    print(f"语料 {len(raw):,} 字节")

    seeds = [0, 1, 2]
    budget = 120.0

    # ---------------- 多种子复现
    print(f"\n=== 多种子复现（每个种子 {budget:.0f} 秒，lr=3e-3）===")
    architectures = {
        "场模型": lambda: WaveFieldLM(FieldConfig(vocab_size=VOCAB, modes=512, layers=4, max_length=LENGTH)),
        "上一版 KernelAttention": lambda: EpicycleLM(
            EpicycleConfig(vocab_size=VOCAB, harmonics=48, layers=3, max_length=LENGTH)
        ),
    }
    summary: dict = {}
    for name, factory in architectures.items():
        print(f"\n  --- {name} ---")
        runs = []
        for seed in seeds:
            row = run(factory, train_data, val_data, device, seed, budget)
            runs.append(row)
            print(
                f"      seed {seed}  {row['steps']:>6,} 步  {row['bits_per_byte']:.4f} bpb  "
                f"峰值 {row['peak_gb']:.3f} GB"
            )
        values = [r["bits_per_byte"] for r in runs]
        summary[name] = {
            "runs": runs,
            "mean_bpb": statistics.mean(values),
            "stdev_bpb": statistics.stdev(values) if len(values) > 1 else 0.0,
            "mean_steps": statistics.mean(r["steps"] for r in runs),
            "mean_peak_gb": statistics.mean(r["peak_gb"] for r in runs),
        }
        print(
            f"      均值 {summary[name]['mean_bpb']:.4f} ± {summary[name]['stdev_bpb']:.4f} bpb"
            f"   平均 {summary[name]['mean_steps']:,.0f} 步"
        )

    print("\n=== 多种子汇总 ===")
    print(f"  {'架构':<24}{'均值 bpb':>12}{'标准差':>10}{'平均步数':>12}{'峰值GB':>10}")
    for name, row in summary.items():
        print(
            f"  {name:<24}{row['mean_bpb']:>12.4f}{row['stdev_bpb']:>10.4f}"
            f"{row['mean_steps']:>12,.0f}{row['mean_peak_gb']:>10.3f}"
        )
    if len(summary) == 2:
        field, old = summary["场模型"], summary["上一版 KernelAttention"]
        gap = old["mean_bpb"] - field["mean_bpb"]
        pooled = math.sqrt(field["stdev_bpb"] ** 2 + old["stdev_bpb"] ** 2)
        separated = gap > 2 * pooled if pooled > 0 else gap > 0
        print(
            f"\n  均值差 {gap:+.4f} bpb，合并标准差 {pooled:.4f}"
            f"  ⟹ 两个分布{'明显分开' if separated else '**未能分开**，差异可能来自随机性'}"
        )

    # ---------------- 场模型消融
    print("\n=== 场模型消融（每个配置 90 秒，单种子）===")
    print(f"  {'modes':>7}{'layers':>8}{'参数':>12}{'步数':>9}{'bpb':>10}")
    ablation = []
    for modes in (128, 256, 512):
        for layers in (2, 4):
            factory = lambda m=modes, l=layers: WaveFieldLM(
                FieldConfig(vocab_size=VOCAB, modes=m, layers=l, max_length=LENGTH)
            )
            row = run(factory, train_data, val_data, device, seed=0, budget=90.0)
            ablation.append({"modes": modes, "layers": layers, **row})
            print(
                f"  {modes:>7}{layers:>8}{row['parameters']:>12,}{row['steps']:>9,}"
                f"{row['bits_per_byte']:>10.4f}"
            )

    output = Path(__file__).resolve().parent / "runs" / "seeds_ablation.json"
    output.write_text(
        json.dumps({"seeds": summary, "ablation": ablation}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
