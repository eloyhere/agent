"""
储备池 vs 反向传播场模型：同台对比。

要回答的问题：
    1. 固定基底 + 线性读出，质量能不能打？
    2. 训练到底有多快？（储备池是一次线性求解，没有梯度下降）
    3. 种子方差如何？（储备池理论上应当确定）
    4. 内存占用落在 1 GB 预算的什么位置？
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

from reservoir import ReservoirConfig, WaveReservoir, RidgeReadout
from wavefield import FieldConfig, WaveFieldLM

VOCAB = 258
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


def one_hot(tokens: torch.Tensor, vocabulary: int, device: str) -> torch.Tensor:
    return F.one_hot(tokens, vocabulary).float().to(device)


def run_reservoir(train_tokens, val_tokens, modes, steps_per_token, ridge, device, seed=0):
    config = ReservoirConfig(
        modes=modes,
        vocabulary=VOCAB,
        steps_per_token=steps_per_token,
        seed=seed,
    )
    reservoir = WaveReservoir(config, device=device)

    started = time.perf_counter()
    train_states = reservoir.run(train_tokens)
    val_states = reservoir.run(val_tokens)
    simulation_seconds = time.perf_counter() - started

    train_features = reservoir.features(train_states[:-1])
    val_features = reservoir.features(val_states[:-1])

    readout = RidgeReadout(ridge=ridge)
    started = time.perf_counter()
    readout.fit(train_features, one_hot(train_tokens[1:], VOCAB, device))
    fit_seconds = time.perf_counter() - started

    logits = readout.predict(val_features)
    loss = F.cross_entropy(logits, val_tokens[1:].to(device)).item()
    bits = loss / math.log(2)

    result = {
        "bits_per_byte": bits,
        "simulation_seconds": simulation_seconds,
        "fit_seconds": fit_seconds,
        "total_seconds": simulation_seconds + fit_seconds,
        "feature_dim": int(train_features.shape[1]),
        "readout_bytes": readout.parameter_bytes(),
        "field_bytes": reservoir.memory_bytes(),
        "cached_state_bytes": readout.state_bytes,
        "train_tokens": int(train_tokens.numel()),
    }
    del reservoir, readout, train_states, val_states, train_features, val_features
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def run_backprop(train_tokens, val_tokens, modes, layers, device, seconds, lr=3e-3, batch=8, length=256):
    """对等的反向传播基线：同样多的时间，能学到什么。"""
    model = WaveFieldLM(FieldConfig(vocab_size=VOCAB, modes=modes, layers=layers, max_length=length)).to(device)
    parameter_count = model.parameter_count()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    steps = 0
    while time.perf_counter() - started < seconds:
        starts = torch.randint(0, train_tokens.numel() - length - 1, (batch,))
        inputs = torch.stack([train_tokens[s : s + length] for s in starts]).to(device)
        targets = torch.stack([train_tokens[s + 1 : s + length + 1] for s in starts]).to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps += 1
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0

    model.eval()
    total = 0.0
    rounds = 8
    with torch.no_grad():
        for _ in range(rounds):
            starts = torch.randint(0, val_tokens.numel() - length - 1, (batch,))
            inputs = torch.stack([val_tokens[s : s + length] for s in starts]).to(device)
            targets = torch.stack([val_tokens[s + 1 : s + length + 1] for s in starts]).to(device)
            total += F.cross_entropy(model(inputs).reshape(-1, VOCAB), targets.reshape(-1)).item() / math.log(2)
    bits = total / rounds
    del model, optimizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "bits_per_byte": bits,
        "parameters": parameter_count,
        "steps": steps,
        "seconds": elapsed,
        "peak_gb": peak,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备 {device}    随机基线 {BASELINE:.4f} bits/byte")
    raw = gather_corpus(Path(__file__).resolve().parents[2] / "periodicity")
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()

    train_tokens = data[:20000]
    val_tokens = data[20000:22000]
    print(f"训练 {train_tokens.numel():,} token / 验证 {val_tokens.numel():,} token")

    # ---------------- 储备池
    print("\n=== 储备池（固定基底 + ridge 线性读出）===")
    print(f"  {'K':>6}{'步/token':>9}{'特征维':>9}{'模拟(s)':>10}{'拟合(s)':>9}{'bpb':>9}")
    reservoir_rows = []
    for modes, steps_per_token, ridge in [
        (1024, 2, 1e-2),
        (2048, 2, 1e-2),
        (4096, 2, 1e-2),
        (2048, 4, 1e-2),
        (2048, 2, 1e-1),
    ]:
        row = run_reservoir(train_tokens, val_tokens, modes, steps_per_token, ridge, device)
        row.update({"modes": modes, "steps_per_token": steps_per_token, "ridge": ridge})
        reservoir_rows.append(row)
        print(
            f"  {modes:>6}{steps_per_token:>9}{row['feature_dim']:>9}"
            f"{row['simulation_seconds']:>10.1f}{row['fit_seconds']:>9.2f}"
            f"{row['bits_per_byte']:>9.4f}"
        )

    best = min(reservoir_rows, key=lambda r: r["bits_per_byte"])
    print(
        f"\n  最佳配置 K={best['modes']} 步/token={best['steps_per_token']} ridge={best['ridge']}"
        f"  ->  {best['bits_per_byte']:.4f} bpb"
    )
    print(f"  拟合耗时 {best['fit_seconds']:.3f} s（一次线性求解，无梯度下降）")
    total_bytes = best["readout_bytes"] + best["field_bytes"] + best["cached_state_bytes"]
    print(
        f"  内存：场 {best['field_bytes'] / 1024:.0f} KB + 读出 {best['readout_bytes'] / 1024 / 1024:.1f} MB"
        f" + 状态缓存 {best['cached_state_bytes'] / 1024 / 1024:.0f} MB"
    )
    print(f"        合计 {total_bytes / 1024 / 1024:.0f} MB  （1 GB 预算的 {total_bytes / 1024**3 * 100:.1f}%）")

    # ---------------- 反向传播基线，给同样多的时间
    backprop_seconds = best["total_seconds"]
    print(f"\n=== 反向传播场模型（同样给 {backprop_seconds:.0f} 秒）===")
    backprop = run_backprop(train_tokens, val_tokens, modes=512, layers=4, device=device, seconds=backprop_seconds)
    print(f"  参数 {backprop['parameters']:,}   步数 {backprop['steps']:,}")
    print(f"  峰值显存 {backprop['peak_gb']:.3f} GB")
    print(f"  bits/byte {backprop['bits_per_byte']:.4f}")

    print("\n=== 汇总 ===")
    print(f"  {'方法':<26}{'bpb':>10}{'训练秒':>10}{'内存':>14}")
    print(
        f"  {'储备池（固定基底）':<26}{best['bits_per_byte']:>10.4f}"
        f"{best['total_seconds']:>10.1f}{str(round(total_bytes / 1024 / 1024)) + ' MB':>14}"
    )
    print(
        f"  {'反向传播场模型':<26}{backprop['bits_per_byte']:>10.4f}"
        f"{backprop['seconds']:>10.1f}{str(round(backprop['peak_gb'] * 1024)) + ' MB':>14}"
    )
    delta = best["bits_per_byte"] - backprop["bits_per_byte"]
    print(f"\n  质量差 {delta:+.4f} bpb（{'储备池更好' if delta < 0 else '储备池略差' if delta > 0.1 else '基本持平'}）")
    print(f"  训练快 {backprop['seconds'] / best['total_seconds']:.1f}×")

    output = Path(__file__).resolve().parent / "runs" / "reservoir.json"
    output.write_text(
        json.dumps({"reservoir": reservoir_rows, "backprop": backprop, "best": best}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
