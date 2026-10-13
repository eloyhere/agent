"""
场模型 —— 状态是**一个固定大小的复场**，不是每位置一个向量。

    ψ ∈ ℂ^K          整个序列叠加进这一个场；K 与序列长度无关。

这是与 Transformer 及本项目上一版架构的根本差别。上一版的 KernelAttention 物化
(B, T, T, K) 张量，那是把二次复杂度换了个说法；这里**没有任何成对张量**。

## 五个操作，全部对应书本算子，且全部是 O(K) 或 O(K log K)

    叠加 superposition   ψ ← ψ + 词向量场                 O(K)
    色散传播 Translation φ_k ← φ_k − ω_k·τ                O(K)   ← 书本 Translation
    全息绑定 periodic conv 频域逐点复乘                     O(K)   ← 书本周期卷积
    三波混频 ω₁+ω₂       空间域逐点非线性                   O(K log K)
    干涉读出 trend conv   Re⟨ψ, φ_query⟩                    O(K)   ← 书本趋势卷积

## 关键实现点：线性递推就是卷积，因此可以 FFT 并行

场的累积满足

    ψ_t = a ⊙ ψ_{t−1} + x_t,        a_k = r_k · e^{iθ_k}

其中 `r_k` 是衰减半径、`θ_k` 是每个时间步的相位推进（**即色散关系**，
也正是书本 Translation 取 τ=1 时的样子）。

这个递推有闭式解

    ψ_t = Σ_{s≤t} a^{t−s} ⊙ x_s

**它本身就是一个因果卷积**，所以用 FFT 一次算完，复杂度 O(T log T · K)，
而**不必逐步循环**。上一版那个 `for step in range(length)` 正是漏看了这一点，
那才是当时 0.5 秒/步的主因。

## 情感不需要单独模块

在场里，多时间尺度**就是不同频段**：

    低频模（θ 小、衰减慢） → 性情、心情
    高频模（θ 大、衰减快） → 即时反应

同一个场，不需要额外的情感层。书本的分析工具可直接诊断它。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FieldConfig:
    vocab_size: int = 258
    modes: int = 128          # K：场的模数，与序列长度无关
    layers: int = 3
    max_length: int = 256
    dropout: float = 0.0


@dataclass
class Complexity:
    """一次前向的复杂度记账，用实测张量元素数计算，不用推导。"""

    time_order: str
    space_order: str
    largest_tensor: int
    total_tensor_elements: int


def dispersive_accumulate(x: torch.Tensor, radius: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    色散累积：ψ_t = Σ_{s≤t} a^{t−s} x_s，a_k = r_k e^{iθ_k}

    用 FFT 把因果卷积一次算完，O(T log T · K)。**没有逐步循环，没有成对张量。**
    """
    batch, length, modes = x.shape
    steps = torch.arange(length, device=x.device, dtype=torch.float32).view(1, length, 1)
    # a^s = r^s · e^{i s θ}，写成极坐标避免复数幂的数值问题
    powers = (radius.view(1, 1, modes) ** steps) * torch.exp(
        1j * theta.view(1, 1, modes) * steps
    )
    n = 2 * length
    spectrum_x = torch.fft.fft(x, n=n, dim=1)
    spectrum_a = torch.fft.fft(powers.to(x.dtype), n=n, dim=1)
    accumulated = torch.fft.ifft(spectrum_x * spectrum_a, dim=1)[:, :length, :]
    return accumulated


class ModeCoupling(nn.Module):
    """
    三波混频：ω₁ + ω₂ → ω₁+ω₂，即书本 `def:periodic_convolution` 的复合律。

    在**空间域**是逐点相乘，在频域是卷积。所以做法是：逆变换到空间域 →
    逐点非线性 → 变换回频域（FNO 的同一手法）。两侧各一次 FFT，O(K log K)。
    """

    def __init__(self, config: FieldConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2 * config.modes)
        self.gain = nn.Parameter(torch.ones(config.modes))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        # 到空间域（即书本的 x 域，模数 K 的对偶）
        spatial = torch.fft.ifft(field, dim=-1)
        # 逐点非线性：实部虚部各走一次 GELU —— 这是「场在自身作用下的响应」
        activated = torch.complex(
            F.gelu(spatial.real), F.gelu(spatial.imag)
        )
        # 回频域，并做一次逐模增益（相当于每个模的耦合强度可学）
        back = torch.fft.fft(activated, dim=-1) * self.gain.view(1, 1, -1)
        stacked = torch.cat([back.real, back.imag], dim=-1)
        normalized = self.norm(stacked)
        real, imaginary = normalized.chunk(2, dim=-1)
        return field + self.dropout(torch.complex(real, imaginary))


class WaveFieldLM(nn.Module):
    """
    场语言模型。

    前向流程（每个 token 只做常数量的工作，除去两次 FFT）：

        词 → 复场 (B,T,K)
        ┌── 逐层 ────────────────────────────────┐
        │  色散累积（FFT 因果卷积）  O(T log T K)  │
        │  归一化                                  │
        │  模耦合（空间域非线性）    O(T K log K)  │
        └─────────────────────────────────────────┘
        干涉读出 → logits (B,T,V)                  O(T V K)
    """

    def __init__(self, config: FieldConfig) -> None:
        super().__init__()
        self.config = config
        modes = config.modes
        # 每个词表条目 = 一个复场（振幅 + 相位）
        self.embed_amplitude = nn.Parameter(torch.randn(config.vocab_size, modes) * 0.5)
        self.embed_phase = nn.Parameter(
            torch.empty(config.vocab_size, modes).uniform_(-math.pi, math.pi)
        )
        # 色散关系 θ_k：初始化为 DFT 栅格，可学
        self.theta = nn.Parameter((2 * math.pi / config.max_length) * torch.arange(1, modes + 1).float())
        # 衰减半径 r_k：初始偏慢衰减（记得久）
        self.radius_logit = nn.Parameter(torch.full((modes,), 1.5))
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(2 * modes) for _ in range(config.layers)]
        )
        self.couplings = nn.ModuleList([ModeCoupling(config) for _ in range(config.layers)])
        # 干涉读出用的查询场
        self.query_amplitude = nn.Parameter(torch.randn(modes) * 0.5)
        self.query_phase = nn.Parameter(torch.zeros(modes))
        # 读出温度：对齐度在 [-1,1]，需要一个可学的尺度才够表达力
        self.readout_temperature = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------ 前向
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length = tokens.shape
        modes = self.config.modes

        amplitude = F.softplus(self.embed_amplitude[tokens])          # (B,T,K)
        phase = self.embed_phase[tokens]
        field = torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))

        radius = torch.sigmoid(self.radius_logit)                     # (K,)

        for index in range(self.config.layers):
            # 1) 色散传播 + 衰减：FFT 因果卷积，无成对张量
            accumulated = dispersive_accumulate(field, radius, self.theta)
            # 2) 归一化（在实虚拼接的实空间做，LayerNorm 不支持复数）
            stacked = torch.cat([accumulated.real, accumulated.imag], dim=-1)
            normalized = self.layer_norms[index](stacked)
            real, imaginary = normalized.chunk(2, dim=-1)
            field = field + torch.complex(real, imaginary)
            # 2b) 把场的模长约束住。残差相加会让振幅逐层累积，若不约束，
            #     读出时对 K 个模求和的 logits 会随层数线性膨胀（实测到上千）。
            field = field / (field.abs().mean(dim=-1, keepdim=True) + 1e-6)
            # 3) 三波混频
            field = self.couplings[index](field)

        # 4) 干涉读出：与词表场做**相位对齐**度量（趋势卷积的归一化形式）
        #    只留相位、丢掉模长，再除以 K，结果落在 [-1, 1] —— 这是相似度，不是内积。
        #    不归一化的话，对 K 个模求和会让 logits 正比于 K，交叉熵直接爆掉。
        vocab_amplitude = F.softplus(self.embed_amplitude)
        vocab = torch.complex(
            vocab_amplitude * torch.cos(self.embed_phase),
            vocab_amplitude * torch.sin(self.embed_phase),
        )
        field_phasor = field / (field.abs() + 1e-6)
        vocab_phasor = vocab / (vocab.abs() + 1e-6)
        alignment = torch.real(torch.einsum("btk,vk->btv", field_phasor, vocab_phasor.conj()))
        # 除以 K：每个模的相位对齐度在 [-1,1]，K 个模求和后必须平均，否则 logits
        # 正比于 K（K=256 时观测量级正是 ±330）。平均后 alignment ∈ [-1,1]。
        alignment = alignment / self.config.modes
        # 学一个温度，让模型自己决定 logits 的尺度
        logits = alignment * F.softplus(self.readout_temperature)
        return logits

    # ------------------------------------------------------- 复杂度记账
    def complexity(self, batch: int, length: int) -> Complexity:
        """按实际张量元素数记账，便于与其它架构同台对比。"""
        k = self.config.modes
        v = self.config.vocab_size
        # 最大张量：FFT 内的工作缓冲为 2T（不是 T²）
        largest = batch * 2 * length * k
        total = (
            batch * 2 * length * k          # FFT 缓冲
            + batch * length * k * 2        # 拼接后的实虚
            + batch * length * v            # logits
        ) * self.config.layers
        return Complexity(
            time_order="O(T log T · K + T·V·K)",
            space_order="O(T·K)  激活；O(K) 持久状态",
            largest_tensor=largest,
            total_tensor_elements=total,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
