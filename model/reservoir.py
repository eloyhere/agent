"""
全息波场储备池 —— 基底固定不训练，只训一个线性读出。

## 范式的转变

前面的 `wavefield.py` 用反向传播训练一个波场。这条路的问题是实测出来的：
`corr(参数量, bpb) = +0.81` —— **参数越多质量越差**。因为 7.68 MB 语料撑不起容量，
容量根本不是瓶颈。

储备池计算（Jaeger 2001；Maass 2002）给出另一条路：**能力来自动力学，不来自参数**。
一个非线性、高维、有记忆的动力学系统，**线性读出**就能提取输入的复杂函数。
基底完全固定，训练是一次线性求解 —— 没有梯度、没有反向传播、没有 epoch。

前沿在往这个方向收拢：波基神经形态硬件、微波衍射神经网络、光子储备池、
神经场理论、跨频耦合、超维计算。三个领域指向同一件事。

## 与书本的对应（这是本项目的独特之处）

随机储备池的动力学是随便挑的。而书本的算子恰好定义了一个**物理上有意义的波场**：

    色散传播    Translation  φ_k ← φ_k − ω_k·τ     每个模按自身频率积累相位
    三波混频    周期卷积     Ω = ω₁+ω₂            非线性模耦合，Φ=φ₁+φ₂ 即相位匹配
    逐模耗散                 r_k                  慢模＝记忆，快模＝反应
    干涉读出    趋势卷积     Re⟨ψ, conj(φ)⟩       全息读取

**多时间尺度是结构性的，不是硬造的**：ω_k 与 r_k 一起给出从"即时反应"到"长期性情"
的连续谱。情感就是场的低频段 —— 这正是神经科学里的**跨频耦合**。

## 规模

    K = 4096 模，complex64       场本体      32 KB
    读出矩阵 K × 词表             4 MB
    训练状态缓存（可降采样）       最多几百 MB

**远低于 1 GB，而且模型本体只有几 MB。**
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class ReservoirConfig:
    modes: int = 2048          # K
    vocabulary: int = 258
    steps_per_token: int = 2   # 每个 token 演化几步
    spectral_radius: float = 0.95   # 回声状态性质要求 < 1
    input_scale: float = 0.6
    dispersion_low: float = 0.02    # 最慢模的角频率
    dispersion_high: float = 1.2    # 最快模的角频率
    nonlinearity: float = 1.0       # 三波混频强度
    seed: int = 0
    dtype: torch.dtype = torch.complex64


class WaveReservoir:
    """
    固定的波场基底。**没有任何可训练参数。**

    所有量（输入投影、色散关系、耗散率）都在构造时定死，之后永不改变。
    唯一"学到"的东西在 `RidgeReadout` 里。
    """

    def __init__(self, config: ReservoirConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        k = config.modes

        # 输入投影：每个词表条目 -> 一个固定的复相量场。**不训练。**
        phase = torch.rand(config.vocabulary, k, generator=generator) * 2 * math.pi
        magnitude = torch.rand(config.vocabulary, k, generator=generator) * 0.5 + 0.5
        self.input_field = (
            magnitude * torch.exp(1j * phase)
        ).to(device=device, dtype=config.dtype) * config.input_scale

        # 色散关系 ω_k：从慢到快铺开。**不训练。**
        # 慢模（ω 小）负责长程记忆，快模负责即时响应 —— 多时间尺度由此而来。
        self.omega = torch.linspace(
            config.dispersion_low, config.dispersion_high, k, device=device
        )

        # 耗散率 r_k：与 ω_k 反相关，保证慢模记得久。**不训练。**
        # 用 spectral_radius 归一，使 r ∈ [0, spectral_radius]，满足回声状态性质。
        normalized = (self.omega - config.dispersion_low) / (
            config.dispersion_high - config.dispersion_low + 1e-9
        )
        self.radius = config.spectral_radius * (1.0 - 0.65 * normalized)

    @torch.no_grad()
    def run(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        跑完整个序列，返回每一步的场状态。

        tokens: (T,) int64  ->  states: (T, K) complex

        每个 token 的处理：
            1. 以固定投影把 token 写入场（叠加）
            2. 演化若干步：色散传播 → 三波混频 → 耗散
        """
        config = self.config
        k = config.modes
        length = tokens.shape[0]
        field = torch.zeros(k, device=self.device, dtype=config.dtype)
        states = torch.empty(length, k, device=self.device, dtype=config.dtype)

        # 三波混频在空间域做：一次 IFFT 过去、逐点非线性、一次 FFT 回来
        for index in range(length):
            # 1) 写入：书本的「叠加」
            field = field + self.input_field[tokens[index]]

            for _ in range(config.steps_per_token):
                # 2) 色散传播（书本的 Translation）＋ 逐模耗散
                field = field * torch.exp(-1j * self.omega) * self.radius
                # 3) 三波混频（书本周期卷积的 Ω = ω₁+ω₂）
                if config.nonlinearity > 0:
                    spatial = torch.fft.ifft(field)
                    mixed = torch.complex(
                        torch.tanh(spatial.real * 2.0), torch.tanh(spatial.imag * 2.0)
                    )
                    field = field + config.nonlinearity * 0.15 * torch.fft.fft(mixed)

            states[index] = field

        return states

    @torch.no_grad()
    def features(self, states: torch.Tensor) -> torch.Tensor:
        """
        把复场状态摊成实特征。

        除实部虚部外，另外给**模长**与**相位** —— 因为三波混频这类非线性
        主要作用在模长上，线性读出需要见到它才能利用。这是只有一次线性求解的
        读出能做到非线性的关键。
        """
        return torch.cat(
            [states.real, states.imag, states.abs(), torch.angle(states)], dim=-1
        )

    def memory_bytes(self) -> int:
        itemsize = 8 if self.config.dtype == torch.complex64 else 16
        return self.config.modes * itemsize * 2  # 场 + 一个状态快照


class RidgeReadout:
    """
    线性读出 —— **整个系统里唯一被训练的部分**。

    解 W = (XᵀX + λI)⁻¹ XᵀY。一次线性求解，没有梯度、没有反向传播、没有 epoch。

    副产品是**结果确定**：同样的输入永远给同样的 W，没有种子方差。
    """

    def __init__(self, ridge: float = 1e-2) -> None:
        self.ridge = ridge
        self.weights: torch.Tensor | None = None
        self.state_bytes = 0

    def fit(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """features: (N, D)；targets: (N, V) one-hot 或概率。"""
        n, d = features.shape
        self.state_bytes = n * d * 4
        gram = features.T @ features
        gram += self.ridge * torch.eye(d, device=features.device)
        self.weights = torch.linalg.solve(gram, features.T @ targets)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("RidgeReadout.fit must be called first")
        return features @ self.weights

    def parameter_bytes(self) -> int:
        return 0 if self.weights is None else self.weights.numel() * 4


@dataclass
class ReservoirReport:
    bits_per_byte: float
    fit_seconds: float
    feature_dim: int
    readout_bytes: int
    field_bytes: int
    cached_state_bytes: int
    notes: list[str] = field(default_factory=list)
