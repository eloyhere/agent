"""
公平对比：按**同等算力预算**比赛，而不是同等步数。

上一轮的对比是无效的，原因有两个，都是我的设置问题：
  1. 两个架构用了同一个学习率（3e-4），而场模型已知需要更高（3e-3 下 200 步就到 3.74）
  2. 400 步对两者都远不够 —— 都还趴在随机基线 ln(258)=5.553 附近

这一轮改两件事：

  **先各自扫学习率**，再在各自最优学习率下比赛。
  **按墙钟时间给同等预算**，而不是同等步数 —— 因为"用更低的复杂度达到同等效果"
  这个命题的正确形式就是「**同样多的时间里，谁学得更好**」。
  同等步数下比较，等于把场模型的 28× 速度优势白白丢掉。

同时报出两者实际完成的步数，让速度优势以"多学了多少步"的形式显现出来。
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epicycle_lm import Config as EpicycleConfig, EpicycleLM
from wavefield import FieldConfig, WaveFieldLM


def gather_corpus(root: Path) -> bytes:
    chunks: list[bytes] = []
    for path in sorted(root.rglob("*.tex")):
        try:
            chunks.append(path.read_bytes())
        except OSError:
            continue
        chunks.append(b"\n")
    return b"".join(chunks)


def make_batch(data: torch.Tensor, batch: int, length: int, device: str):
    starts = torch.randint(0, data.numel() - length - 1, (batch,))
    inputs = torch.stack([data[s : s + length] for s in starts]).to(device)
    targets = torch.stack([data[s + 1 : s + length + 1] for s in starts]).to(device)
    return inputs, targets


@torch.no_grad()
def evaluate(model, data, batch, length, device, rounds=8) -> float:
    model.eval()
    total = 0.0
    for _ in range(rounds):
        inputs, targets = make_batch(data, batch, length, device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total += loss.item() / math.log(2)
    model.train()
    return total / rounds


def train_budget(model, train_data, batch, length, device, lr, budget_seconds, log_every=200):
    """在墙钟预算内尽量多训。返回 (步数, 用时, 最终训练损失)。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    started = time.perf_counter()
    step = 0
    last = float("nan")
    while True:
        inputs, targets = make_batch(train_data, batch, length, device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = loss.item()
        step += 1
        elapsed = time.perf_counter() - started
        if elapsed >= budget_seconds:
            break
        if step % log_every == 0:
            print(f"      step {step:5d}  loss {last:.4f}  ({elapsed:.0f}s)")
    del optimizer
    return step, time.perf_counter() - started, last


def sweep_learning_rates(name, factory, train_data, val_data, batch, length, device, candidates, steps=250):
    print(f"\n  --- {name} 学习率扫描（每个 {steps} 步）---")
    best = None
    for lr in candidates:
        model = factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        model.train()
        for _ in range(steps):
            inputs, targets = make_batch(train_data, batch, length, device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        bits = evaluate(model, val_data, batch, length, device)
        print(f"      lr {lr:<8} -> bits/byte {bits:.4f}   （随机基线 8.011）")
        if best is None or bits < best[1]:
            best = (lr, bits)
        del model, optimizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"      最优 lr = {best[0]}  ({best[1]:.4f} bits/byte)")
    return best[0]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    corpus_root = Path(__file__).resolve().parents[2] / "periodicity"
    raw = gather_corpus(corpus_root)
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    split = int(data.numel() * 0.98)
    train_data, val_data = data[:split], data[split:]

    batch, length, vocabulary = 8, 256, 258
    print(f"语料 {len(raw):,} 字节   批 {batch}   长 {length}")
    print(f"随机基线 = ln(258)/ln(2) = {math.log(vocabulary) / math.log(2):.4f} bits/byte")

    # 参数量对齐到约 234K（与上一版相当）
    architectures = {
        "场模型": lambda: WaveFieldLM(
            FieldConfig(vocab_size=vocabulary, modes=512, layers=4, max_length=length)
        ),
        "上一版 KernelAttention": lambda: EpicycleLM(
            EpicycleConfig(vocab_size=vocabulary, harmonics=48, layers=3, max_length=length)
        ),
    }
    print("\n参数量：")
    for name, factory in architectures.items():
        probe = factory()
        print(f"  {name:<24} {probe.parameter_count():>10,}")
        del probe
    if device == "cuda":
        torch.cuda.empty_cache()

    candidates = [3e-4, 1e-3, 3e-3]
    best_lr: dict[str, float] = {}
    print("\n=== 第一步：各自扫学习率 ===")
    for name, factory in architectures.items():
        best_lr[name] = sweep_learning_rates(
            name, factory, train_data, val_data, batch, length, device, candidates
        )

    # 同等墙钟预算
    budget = 240.0
    print(f"\n=== 第二步：同等算力预算 {budget:.0f} 秒 ===")
    results: dict = {}
    for name, factory in architectures.items():
        print(f"\n  --- {name}（lr={best_lr[name]}）---")
        model = factory().to(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        steps, elapsed, _ = train_budget(
            model, train_data, batch, length, device, best_lr[name], budget
        )
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
        bits = evaluate(model, val_data, batch, length, device)
        print(f"      完成 {steps:,} 步 / {elapsed:.1f}s   峰值显存 {peak:.3f} GB")
        print(f"      验证 bits/byte {bits:.4f}")
        results[name] = {
            "parameters": model.parameter_count(),
            "learning_rate": best_lr[name],
            "steps": steps,
            "seconds": elapsed,
            "ms_per_step": elapsed / steps * 1000,
            "peak_gb": peak,
            "bits_per_byte": bits,
        }
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n=== 汇总（同等 {budget:.0f} 秒算力预算）===")
    print(f"  {'架构':<24}{'参数':>10}{'lr':>9}{'步数':>10}{'ms/步':>9}{'峰值GB':>9}{'bpb':>9}")
    for name, row in results.items():
        print(
            f"  {name:<24}{row['parameters']:>10,}{row['learning_rate']:>9.0e}"
            f"{row['steps']:>10,}{row['ms_per_step']:>9.1f}{row['peak_gb']:>9.3f}"
            f"{row['bits_per_byte']:>9.4f}"
        )

    baseline = math.log(vocabulary) / math.log(2)
    print(f"\n  随机基线 {baseline:.4f} bits/byte")
    for name, row in results.items():
        gain = baseline - row["bits_per_byte"]
        print(f"  {name:<24} 相比随机基线降低 {gain:.4f} bpb")

    if len(results) == 2:
        field = results["场模型"]
        old = results["上一版 KernelAttention"]
        print(
            f"\n  同等算力下场模型多学了 {field['steps'] / old['steps']:.1f}× 的步数；"
            f"质量差 {field['bits_per_byte'] - old['bits_per_byte']:+.4f} bpb"
        )

    output = Path(__file__).resolve().parent / "runs" / "fair_compare.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
