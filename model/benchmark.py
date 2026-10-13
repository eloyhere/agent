"""
复杂度实测：上一版（KernelAttention）对抗新版（场模型）。

**为什么必须实测而不是推导：** 上一版我也"推导"过它的复杂度，结论是错的——
我以为省了，实际上物化 (B,T,T,K) 把二次复杂度原封不动搬了过来。所以这一轮
用 `torch.cuda.max_memory_allocated()` 与墙钟时间量，并把 T 从 64 扫到 1024，
看增长趋势到底是线性还是二次。

判据：**若空间是 O(T·K)，峰值显存随 T 线性增长；若是 O(T²·K)，则二次增长。**
把实测值除以理论阶数，比值应当近似常数——这就是可证伪的形式。
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epicycle_lm import Config as EpicycleConfig, EpicycleLM   # 上一版
from wavefield import FieldConfig, WaveFieldLM                  # 新版


def measure(model, vocabulary: int, batch: int, length: int, device: str, warmup: int = 2, repeat: int = 3):
    """量一次前向+反向的峰值显存与耗时。OOM 时返回 None 而不是崩溃。"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randint(0, vocabulary, (batch, length), device=device)

    def one_step() -> None:
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    try:
        for _ in range(warmup):
            one_step()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(repeat):
            one_step()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / repeat
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
        return {"seconds": elapsed, "peak_gb": peak}
    except torch.cuda.OutOfMemoryError:
        return None
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            return None
        raise
    finally:
        del optimizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(f"  {properties.name}  {properties.total_memory / 1024**3:.1f} GB")

    # 参数量对齐：让两者规模相当，比较才有意义
    vocab = 258
    lengths = [64, 128, 256, 512, 1024]
    batch = 8

    configurations = {
        "上一版 KernelAttention": lambda: EpicycleLM(
            EpicycleConfig(vocab_size=vocab, harmonics=48, layers=3, max_length=1024)
        ),
        "新版 场模型": lambda: WaveFieldLM(
            FieldConfig(vocab_size=vocab, modes=128, layers=3, max_length=1024)
        ),
    }

    print(f"\n批大小 {batch}，参数量对齐情况：")
    for name, factory in configurations.items():
        model = factory()
        print(f"  {name:24} {model.parameter_count():>10,} 参数")
        del model
    if device == "cuda":
        torch.cuda.empty_cache()

    results: dict = {}
    for name, factory in configurations.items():
        print(f"\n=== {name} ===")
        print(f"  {'T':>6} {'峰值显存(GB)':>14} {'每步(ms)':>12} {'显存/T':>12} {'显存/T²':>12}")
        rows = []
        for length in lengths:
            model = factory().to(device)
            outcome = measure(model, vocab, batch, length, device)
            if outcome is None:
                print(f"  {length:>6} {'显存不足':>14} {'—':>12} {'—':>12} {'—':>12}")
                rows.append({"length": length, "oom": True})
            else:
                per_t = outcome["peak_gb"] / length
                per_t2 = outcome["peak_gb"] / (length * length)
                print(
                    f"  {length:>6} {outcome['peak_gb']:>14.3f} {outcome['seconds'] * 1000:>12.1f} "
                    f"{per_t:>12.6f} {per_t2:>12.8f}"
                )
                rows.append(
                    {
                        "length": length,
                        "peak_gb": outcome["peak_gb"],
                        "seconds": outcome["seconds"],
                        "per_t": per_t,
                        "per_t2": per_t2,
                    }
                )
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        results[name] = rows

    # ---- 判定：线性还是二次
    print("\n=== 增长阶数判定 ===")
    print("  做法：把实测峰值显存分别除以 T 与 T²，看哪一个的比值更接近常数。")
    for name, rows in results.items():
        usable = [r for r in rows if not r.get("oom")]
        if len(usable) < 3:
            print(f"  {name}: 可用数据点不足")
            continue
        first, last = usable[0], usable[-1]
        ratio_t = last["per_t"] / first["per_t"]
        ratio_t2 = last["per_t2"] / first["per_t2"]
        length_ratio = last["length"] / first["length"]
        # 完美线性时 per_t 比值 = 1；完美二次时 per_t2 比值 = 1
        verdict = "线性 O(T)" if abs(ratio_t - 1) < abs(ratio_t2 - 1) else "二次 O(T²)"
        print(
            f"  {name:24} T 增长 {length_ratio:.0f}×  →  "
            f"显存/T 变化 {ratio_t:5.2f}×，显存/T² 变化 {ratio_t2:5.2f}×   ⟹ {verdict}"
        )

    output = Path(__file__).resolve().parent / "runs" / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已保存 {output}")


if __name__ == "__main__":
    main()
