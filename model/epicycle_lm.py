"""
周转圆语言模型 —— 按周期数学构建的序列模型。

与 Transformer 的根本差别在于**状态是什么**。Transformer 的状态是一个没有内在
时间结构的实向量；这里的每个状态都是一个**周转圆组**

    E(x) = Σ_k A_k · e^{i(ω_k x + φ_k)}

即书本 `def:eaf_agent` 的 CEAF。于是书本的算子可以直接当作网络层使用，而不是
先造一个网络再回头解释它。

三层结构，逐一对应书本的算子：

  1. EpicycleEmbedding      每个词表条目是一个周转圆组 (A, ω, φ)
  2. EpicycleBlock
       · Translation        位置赋值：φ_k ← φ_k − ω_k·n
                            **这正是 RoPE**。书本的 Translation 算子与当今主流 LLM
                            的旋转位置编码是同一个东西，这里直接用它。
       · Resonator          线性路径：复对角递推 h_n = α ⊙ h_{n−1} + c_n，
                            α = r·e^{iθ} 为衰减旋转。这就是一 bank 的谐振子，
                            也是 `def:periodic_convolution` 在序列上的实现。
                            O(n)，且它对**长程周期结构**天生敏感 —— 正是我们想要的。
       · KernelAttention    二次路径：用书本 `prop:twelve_to_six` 的六个相似核
                            在相位差上打分，再 softmax 加权。这就是注意力，
                            只不过度量换成了相位。
  3. CEAF 读出 + 趋势卷积投影   logits_v = Σ_k A_k·A^v_k·cos(φ_k − φ^v_k)
                            即 `def:tendency_convolution`，且**权重与嵌入共享**。

关于情感：本文件只做语言模型。情感状态层是同一个数学对象（一个 Waveform），
放在 affective.py 里，它同时是模型的状态与可被项目自己的分析器诊断的对象。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    """模型配置。"""

    vocab_size: int = 258            # 256 字节 + BOS + EOS
    harmonics: int = 64              # 每个位置携带的周转圆个数 K
    layers: int = 4
    max_length: int = 256
    attention_heads: int = 4
    dropout: float = 0.1
    # 相位注意力的倍角，取书本 prop:twelve_to_six 里的 1/2、1、2 三档
    kernel_scales: tuple = (0.5, 1.0, 2.0)
    # 谐振子衰减半径的初值；<1 保证稳定，学习的是「记多久」
    decay_init: float = 0.9


class EpicycleEmbedding(nn.Module):
    """
    每个词表条目 = 一个周转圆组。

    存的是 (振幅对数, 相位, 角频率) 三组参数，前向时合成复数分量
    c[k] = softplus(a[k]) · e^{i·p[k]}。

    角频率 ω 是**学出来的**，初始化在第 k 个谐波 2πk/L 上。这一步让模型可以自己
    决定「用哪些周期去描述一个字节」，而不是由我们固定成 DFT 的等距栅格。
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        k = config.harmonics
        self.amplitude_logit = nn.Parameter(torch.zeros(config.vocab_size, k))
        self.phase = nn.Parameter(torch.empty(config.vocab_size, k).uniform_(-math.pi, math.pi))
        # 初始化为 DFT 的等距栅格，随后可自由移动
        base = (2 * math.pi / config.max_length) * torch.arange(1, k + 1, dtype=torch.float32)
        self.omega = nn.Parameter(base.unsqueeze(0).repeat(config.vocab_size, 1))

    def forward(self, tokens: torch.Tensor):
        """tokens: (B, T) int64 -> (amplitude, omega, phase) 各 (B, T, K)。"""
        amplitude = F.softplus(self.amplitude_logit[tokens])
        omega = self.omega[tokens]
        phase = self.phase[tokens]
        return amplitude, omega, phase


def translate(phase: torch.Tensor, omega: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    书本的 Translation 算子：把位置编码成依赖频率的相位旋转

        φ_k ← φ_k − ω_k · n

    **这一段就是 RoPE。** 书本在 `def:translation_operator` 给出它，
    并在 `ex:phase_functor_action` 里写成了 e^{iθ}f(t − θT/2π)。
    """
    # positions 需按 (1, T, 1) 广播到 (B, T, K)；写成 (T, 1, 1) 会与 batch 维冲突
    return phase - omega * positions.view(1, -1, 1)


class Resonator(nn.Module):
    """
    线性混合路径：一 bank 的阻尼旋转子

        h_n = α ⊙ h_{n−1} + c_n,      α = r · e^{iθ}

    每个谐波有自己的 (r, θ)，由网络学习。r → 1 表示该频率几乎不衰减，
    于是这个通道负责**长程周期**；r 小则只管近处。

    这既是 `def:periodic_convolution`（频率与相位相加、振幅相乘）在序列上的
    实现，也与状态空间模型（S4 一类）的复对角形式同构 —— 而那类模型的动机
    恰恰是长程周期结构。O(T) 复杂度。
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        k = config.harmonics
        # 用 logit 参数化半径，sigmoid 保证 (0,1) 稳定
        radius = torch.full((k,), math.log(config.decay_init / (1 - config.decay_init)))
        self.radius_logit = nn.Parameter(radius)
        self.theta = nn.Parameter(torch.zeros(k))
        self.out_proj = nn.Linear(2 * k, 2 * k)

    def forward(self, real: torch.Tensor, imaginary: torch.Tensor) -> torch.Tensor:
        """real/imaginary: (B, T, K) -> (B, T, 2K)"""
        radius = torch.sigmoid(self.radius_logit)                 # (K,)
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)

        batch, length, k = real.shape
        hr = real.new_zeros(batch, k)
        hi = real.new_zeros(batch, k)
        outputs = []
        for step in range(length):
            cr = real[:, step, :]
            ci = imaginary[:, step, :]
            # 复数乘 α：h ← (r e^{iθ}) h
            nr = radius * (cos_t * hr - sin_t * hi) + cr
            ni = radius * (sin_t * hr + cos_t * hi) + ci
            hr, hi = nr, ni
            outputs.append(torch.cat([hr, hi], dim=-1))
        stacked = torch.stack(outputs, dim=1)                     # (B, T, 2K)
        return self.out_proj(stacked)


class KernelAttention(nn.Module):
    """
    二次混合路径：相位核注意力。

    书本 `prop:twelve_to_six` 给了六个相似核。这里取其中三档倍角
    （1/2、1、2），组成一个可学习的核组合：

        score[i,j] = Σ_h w_h · Σ_k A_i,k · A_j,k · cos(s_h · (φ_i,k − φ_j,k))

    这与标准注意力同构 —— 只不过点积换成了**相位差的余弦**，于是「谁该注意谁」
    由相位对齐程度决定，而不是实向量内积。因果掩码照常。
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.heads = config.attention_heads
        self.kernel_weights = nn.Parameter(torch.zeros(len(config.kernel_scales), self.heads))
        self.value_proj = nn.Linear(2 * config.harmonics, 2 * config.harmonics)
        self.output_proj = nn.Linear(2 * config.harmonics, 2 * config.harmonics)

    def forward(
        self,
        real: torch.Tensor,
        imaginary: torch.Tensor,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, k = real.shape
        # 相位差 (B, H, T, T, K) 会爆显存；先在 K 上做内积再对核求和
        delta = phase.unsqueeze(2) - phase.unsqueeze(1)            # (B, T, T, K)
        weight = amplitude.unsqueeze(2) * amplitude.unsqueeze(1)   # (B, T, T, K)
        scores = real.new_zeros(batch, self.heads, length, length)
        for index, scale in enumerate(self.config.kernel_scales):
            kernel = torch.cos(scale * delta)                      # (B, T, T, K)
            contribution = (weight * kernel).sum(dim=-1)           # (B, T, T)
            scores = scores + self.kernel_weights[index].view(1, self.heads, 1, 1) * contribution.unsqueeze(1)

        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=real.device), diagonal=1)
        scores = scores.masked_fill(mask.view(1, 1, length, length), float("-inf"))
        attention = torch.softmax(scores / math.sqrt(k), dim=-1)

        values = self.value_proj(torch.cat([real, imaginary], dim=-1))   # (B, T, 2K)
        # 每个头共享同一组值，用头的平均注意力加权（保持参数量可控）
        average = attention.mean(dim=1)                                  # (B, T, T)
        mixed = torch.einsum("bij,bjd->bid", average, values)
        return self.output_proj(mixed)


class EpicycleBlock(nn.Module):
    """一层：Translation 已在嵌入处做过，这里做两条混合路径 + 前馈。"""

    def __init__(self, config: Config) -> None:
        super().__init__()
        k = config.harmonics
        self.resonator = Resonator(config)
        self.attention = KernelAttention(config)
        self.norm1 = nn.LayerNorm(2 * k)   # [real, imaginary] 与两条路径的输出都是 2K
        self.norm2 = nn.LayerNorm(2 * k)
        self.feed_forward = nn.Sequential(
            nn.Linear(2 * k, 4 * k),
            nn.GELU(),
            nn.Linear(4 * k, 2 * k),
        )
        # 两条路径的混合比，初始偏向谐振子（便宜且擅长长程）
        self.blend = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, state, real, imaginary, amplitude, phase):
        linear = self.resonator(real, imaginary)                     # (B, T, 2K)
        quadratic = self.attention(real, imaginary, amplitude, phase)
        ratio = torch.sigmoid(self.blend)
        mixed = ratio * linear + (1 - ratio) * quadratic
        hidden = self.norm1(torch.cat([real, imaginary], dim=-1) + self.dropout(mixed))
        hidden = self.norm2(hidden + self.dropout(self.feed_forward(hidden)))
        real_out, imaginary_out = hidden.chunk(2, dim=-1)
        return hidden, real_out, imaginary_out, amplitude, phase


class EpicycleLM(nn.Module):
    """完整的周转圆语言模型。"""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embedding = EpicycleEmbedding(config)
        self.blocks = nn.ModuleList([EpicycleBlock(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(2 * config.harmonics)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tokens: torch.Tensor):
        batch, length = tokens.shape
        amplitude, omega, phase = self.embedding(tokens)
        positions = torch.arange(length, device=tokens.device, dtype=torch.float32)
        # Translation：位置即相位旋转
        phase = translate(phase, omega, positions)

        real = amplitude * torch.cos(phase)
        imaginary = amplitude * torch.sin(phase)

        hidden = torch.cat([real, imaginary], dim=-1)
        for block in self.blocks:
            hidden, real, imaginary, amplitude, phase = block(
                hidden, real, imaginary, amplitude, phase
            )
        hidden = self.final_norm(hidden)
        real, imaginary = hidden.chunk(2, dim=-1)

        # 把隐状态重新读成周转圆组：振幅取模，相位取辐角
        state_amplitude = torch.sqrt(real * real + imaginary * imaginary + 1e-8)
        state_phase = torch.atan2(imaginary, real)

        # 词表投影 = 趋势卷积 def:tendency_convolution，权重与嵌入共享
        #   logits_v = Σ_k A_k · A^v_k · cos(φ_k − φ^v_k)
        vocab_amplitude = F.softplus(self.embedding.amplitude_logit)   # (V, K)
        vocab_phase = self.embedding.phase                             # (V, K)
        delta = state_phase.unsqueeze(2) - vocab_phase.view(1, 1, -1, self.config.harmonics)
        weights = state_amplitude.unsqueeze(2) * vocab_amplitude.view(1, 1, -1, self.config.harmonics)
        logits = (weights * torch.cos(delta)).sum(dim=-1)              # (B, T, V)
        return logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
