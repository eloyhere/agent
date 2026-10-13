"""
在书稿语料上训练周转圆语言模型。

用法（在装了 torch 的 Python 下）：

    python model/train.py --data <书稿目录> --steps 4000 --out model/runs/base

语料只用书稿的 LaTeX 源。规模上必须说清：书稿双语全文约 6 MB ≈ 200 万字节，
而按 Chinchilla 比例，一个 5000 万参数的模型需要约 10 亿 token。**差约 500 倍**。
所以这里的目标不是训出会对话的模型，而是回答一个具体问题：

    「用周期数学搭的架构，在真数据上能不能训得动、学到结构？」

报告里会同时给出损失曲线与**对学到的角频率的诊断** —— 后者尤其重要：
如果模型真的在用周期结构，学出来的 ω 分布应当出现可读的聚集，而不是一团噪声。
那是用项目自己的数学去检查项目自己的模型。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from epicycle_lm import Config, EpicycleLM

BOS = 256
EOS = 257


def gather_corpus(root: Path, limit_files: int | None = None) -> bytes:
    """把书稿的 .tex 全部读成一段字节序列。"""
    files = sorted(root.rglob("*.tex"))
    if limit_files is not None:
        files = files[:limit_files]
    chunks: list[bytes] = []
    for path in files:
        try:
            chunks.append(path.read_bytes())
        except OSError:
            continue
        chunks.append(b"\n")
    return b"".join(chunks)


def make_batches(data: torch.Tensor, batch_size: int, length: int, device: str):
    """随机采样定长片段。"""
    maximum = data.numel() - length - 1
    starts = torch.randint(0, maximum, (batch_size,))
    inputs = torch.stack([data[s : s + length] for s in starts]).to(device)
    targets = torch.stack([data[s + 1 : s + length + 1] for s in starts]).to(device)
    return inputs, targets


def diagnose_omegas(model: EpicycleLM) -> dict:
    """
    对学到的角频率做周期性诊断。

    这是本项目的自指之处：**用项目自己的数学检查项目自己的模型**。
    把每个谐波通道的 ω 收集起来，看它是否形成了可读的结构（若干聚集），
    还是散成一团。用变异系数与直方图峰值度来量化。
    """
    with torch.no_grad():
        omega = model.embedding.omega.detach().float().cpu()
        # 每个谐波通道在词表上的均值，代表该通道的整体频率位置
        per_channel = omega.mean(dim=0)
        per_channel, _ = torch.sort(per_channel)
        spread = per_channel.std().item()
        mean = per_channel.mean().item()
        # 相邻通道之间的间隔是否均匀：均匀则说明仍停在 DFT 栅格上没怎么动
        gaps = (per_channel[1:] - per_channel[:-1]) if per_channel.numel() > 1 else torch.zeros(1)
        gap_uniformity = (gaps.std() / (gaps.mean() + 1e-9)).item() if gaps.numel() > 0 else 0.0
        return {
            "omega_mean": mean,
            "omega_std": spread,
            "omega_min": per_channel.min().item(),
            "omega_max": per_channel.max().item(),
            "gap_uniformity": gap_uniformity,
            "omega_sample": per_channel[:16].tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(Path(__file__).resolve().parents[2] / "periodicity"))
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "runs" / "base"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--harmonics", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  torch {torch.__version__}  CUDA {torch.version.cuda}")

    corpus_root = Path(args.data)
    print(f"\n读取语料: {corpus_root}")
    raw = gather_corpus(corpus_root, args.limit_files)
    print(f"  字节数: {len(raw):,}  ({len(raw)/1024/1024:.2f} MB)")
    if len(raw) < 10000:
        raise SystemExit("语料太小，检查 --data 路径")

    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    split = int(data.numel() * 0.98)
    train_data, val_data = data[:split], data[split:]
    print(f"  训练 {train_data.numel():,} 字节 / 验证 {val_data.numel():,} 字节")

    config = Config(harmonics=args.harmonics, layers=args.layers, max_length=args.length)
    model = EpicycleLM(config).to(device)
    print(f"\n参数量: {model.parameter_count():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    started = time.time()

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = make_batches(train_data, args.batch_size, args.length, device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(8):
                    vi, vt = make_batches(val_data, args.batch_size, args.length, device)
                    vl = model(vi)
                    val_losses.append(
                        F.cross_entropy(vl.reshape(-1, vl.size(-1)), vt.reshape(-1)).item()
                    )
                val_loss = sum(val_losses) / len(val_losses)
            model.train()
            elapsed = time.time() - started
            record = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "perplexity": round(math.exp(val_loss), 2),
                "bits_per_byte": round(val_loss / math.log(2), 4),
                "seconds": round(elapsed, 1),
                "vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else 0.0,
            }
            history.append(record)
            print(
                f"  step {step:5d}  train {record['train_loss']:.4f}  "
                f"val {record['val_loss']:.4f}  ppl {record['perplexity']:.1f}  "
                f"bpb {record['bits_per_byte']:.3f}  {elapsed:.0f}s  "
                f"vram {record['vram_gb']:.2f}GB"
            )

    # ---- 训练后的诊断
    print("\n=== 对学到的角频率做周期性诊断 ===")
    diagnosis = diagnose_omegas(model)
    for key, value in diagnosis.items():
        if key != "omega_sample":
            print(f"  {key:16} {value:.6f}" if isinstance(value, float) else f"  {key:16} {value}")
    print(f"  omega_sample     {[round(v, 4) for v in diagnosis['omega_sample'][:8]]}")
    print(
        "\n  读法：gap_uniformity 接近 0 说明各通道频率仍停在初始的等距栅格上；"
        "\n        明显大于 0 说明模型把频率挪动了 —— 它在自己选择要用哪些周期。"
    )

    # ---- 生成样本
    print("\n=== 生成样本（贪心与前 3 采样）===")
    model.eval()
    prompt = b"\\section{"
    tokens = torch.tensor([[b for b in prompt]], dtype=torch.long, device=device)
    with torch.no_grad():
        for temperature in (0.0, 0.8):
            sequence = tokens.clone()
            for _ in range(200):
                logits = model(sequence)[:, -1, :]
                if temperature == 0.0:
                    nxt = logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    nxt = torch.multinomial(probs, 1)
                sequence = torch.cat([sequence, nxt], dim=1)
            text = bytes(int(b) for b in sequence[0].tolist()[:256] if int(b) < 256)
            label = "贪心" if temperature == 0.0 else "采样 T=0.8"
            print(f"\n  [{label}]\n  {text[:220]!r}")

    torch.save(
        {"config": config.__dict__, "state_dict": model.state_dict(), "history": history, "diagnosis": diagnosis},
        out_dir / "checkpoint.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n已保存到 {out_dir}")


if __name__ == "__main__":
    main()
