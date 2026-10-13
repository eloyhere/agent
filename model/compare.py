"""
同台质量对比：上一版（KernelAttention）对抗新版（场模型）。

复杂度已在 benchmark.py 里用实测钉死（线性 vs 二次）。这里回答另一半问题：

    **用更低的复杂度，语言建模质量会不会变差？**

做法：同一份书稿语料、同样的训练步数与批大小、参数量尽量对齐，
比 bits/byte（字节级，与词表无关，比困惑度更干净）。

同时记录每一步的墙钟时间与峰值显存 —— 因为如果质量相当而速度快 50 倍，
那结论就不是"差不多"，而是"严格更优"。
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


def evaluate(model, data: torch.Tensor, batch: int, length: int, device: str, rounds: int = 6) -> float:
    """返回 bits/byte。"""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(rounds):
            starts = torch.randint(0, data.numel() - length - 1, (batch,))
            inputs = torch.stack([data[s : s + length] for s in starts]).to(device)
            targets = torch.stack([data[s + 1 : s + length + 1] for s in starts]).to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total += loss.item() / math.log(2)
    return total / rounds


def train_one(name, factory, train_data, val_data, steps, batch, length, device, lr=3e-4):
    print(f"\n=== {name} ===")
    model = factory().to(device)
    parameters = model.parameter_count()
    print(f"  参数量 {parameters:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    model.train()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    last_loss = float("nan")
    for step in range(1, steps + 1):
        starts = torch.randint(0, train_data.numel() - length - 1, (batch,))
        inputs = torch.stack([train_data[s : s + length] for s in starts]).to(device)
        targets = torch.stack([train_data[s + 1 : s + length + 1] for s in starts]).to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = loss.item()
        if step % max(1, steps // 5) == 0:
            print(f"  step {step:5d}  train {last_loss:.4f}")

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    bits = evaluate(model, val_data, batch, length, device)

    print(f"  训练用时 {elapsed:.1f} s（{elapsed / steps * 1000:.1f} ms/步）")
    print(f"  峰值显存 {peak:.3f} GB")
    print(f"  验证 bits/byte {bits:.4f}   困惑度(2^bpb) {2 ** min(bits, 60):,.1f}")

    del model, optimizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "parameters": parameters,
        "seconds": elapsed,
        "ms_per_step": elapsed / steps * 1000,
        "peak_gb": peak,
        "bits_per_byte": bits,
    }


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
    print(f"语料 {len(raw):,} 字节   训练 {train_data.numel():,} / 验证 {val_data.numel():,}")

    steps = 400
    batch = 8
    length = 256
    vocabulary = 258

    print(f"\n配置：{steps} 步，批 {batch}，长 {length}，T=256（上一版在此需要约 2.6 GB）")

    results = {}
    # 场模型放在前面：若旧版爆显存，至少新版的数字已经拿到
    results["新版 场模型"] = train_one(
        "新版 场模型（modes=256, layers=4）",
        lambda: WaveFieldLM(FieldConfig(vocab_size=vocabulary, modes=256, layers=4, max_length=length)),
        train_data, val_data, steps, batch, length, device,
    )
    results["上一版 KernelAttention"] = train_one(
        "上一版 KernelAttention（harmonics=48, layers=3）",
        lambda: EpicycleLM(EpicycleConfig(vocab_size=vocabulary, harmonics=48, layers=3, max_length=length)),
        train_data, val_data, steps, batch, length, device,
    )

    print("\n=== 汇总 ===")
    print(f"  {'架构':<26}{'参数':>10}{'ms/步':>10}{'峰值GB':>10}{'bits/byte':>12}")
    for name, row in results.items():
        print(
            f"  {name:<26}{row['parameters']:>10,}{row['ms_per_step']:>10.1f}"
            f"{row['peak_gb']:>10.3f}{row['bits_per_byte']:>12.4f}"
        )

    if len(results) == 2:
        field = results["新版 场模型"]
        old = results["上一版 KernelAttention"]
        speed = old["ms_per_step"] / field["ms_per_step"]
        memory = old["peak_gb"] / field["peak_gb"] if field["peak_gb"] > 0 else float("inf")
        delta = field["bits_per_byte"] - old["bits_per_byte"]
        print(f"\n  速度：场模型快 {speed:.1f}×")
        print(f"  显存：场模型省 {memory:.1f}×")
        print(f"  质量：bits/byte 差 {delta:+.4f}（{'场模型更好' if delta < 0 else '场模型略差' if delta > 0.02 else '基本持平'}）")

    output = Path(__file__).resolve().parent / "runs" / "compare.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
