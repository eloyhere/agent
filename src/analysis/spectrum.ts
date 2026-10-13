import { realDft } from "../mathematics/fft";
import { demean } from "./periods";

/**
 * 谱估计。
 *
 * 一切遵循书本的归一化 DFT 约定
 *   `f̂(k) = (1/T) Σ_{n<T} f(n)·e^{-2πikn/T}`。
 *
 * **修正过的公式（验证报告缺陷 7）**
 *
 * 加窗频谱在频域是**卷积**，不是「窗核 × Dirichlet 核」的乘积。正确的只有下面
 * 这一个合成核 `windowedKernel`。书本原来的乘积写法相对误差达 1.21（Hann 窗），
 * 两版都已修正。
 */

/** 周期图。频率单位为「周期 / 采样」，功率之和等于样本方差（Parseval）。 */
export interface Periodogram {
    readonly frequencies: Float64Array;
    readonly power: Float64Array;
    readonly length: number;
}

export const periodogram: (
    values: ArrayLike<number>,
    options?: { detrend?: boolean }
) => Periodogram = (
    values: ArrayLike<number>,
    options: { detrend?: boolean } = {}
): Periodogram => {
    const detrend: boolean = options.detrend ?? true;
    const n: number = values.length;
    const working = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
        working[i] = values[i];
    }
    if (detrend) {
        // 同时去掉均值与线性趋势：缓慢漂移否则会霸占低频端
        const m: number = working.reduce((a, b) => a + b, 0) / n;
        let sx = 0;
        let sxx = 0;
        let sxy = 0;
        for (let i = 0; i < n; i += 1) {
            sx += i;
            sxx += i * i;
            sxy += i * (working[i] - m);
        }
        const denominator: number = n * sxx - sx * sx;
        const slope: number = denominator !== 0 ? (n * sxy - sx * 0) / denominator : 0;
        const intercept: number = m - (slope * sx) / n;
        for (let i = 0; i < n; i += 1) {
            working[i] -= intercept + slope * i;
        }
    }

    const transformed = realDft(working);
    const length: number = transformed.real.length;
    const frequencies = new Float64Array(length);
    const power = new Float64Array(length);
    for (let k = 0; k < length; k += 1) {
        frequencies[k] = k / n;
        power[k] =
            (transformed.real[k] * transformed.real[k] +
                transformed.imaginary[k] * transformed.imaginary[k]) /
            (n * n);
    }
    // 把负频率折回正频率（直流与 Nyquist 除外）
    const foldLimit: number = length - (n % 2 === 0 ? 1 : 0);
    for (let k = 1; k < foldLimit; k += 1) {
        power[k] *= 2;
    }
    return { frequencies, power, length };
};

/** 峰值（三点抛物线插值，给出亚栅格分辨率）。 */
export interface RefinedPeak {
    readonly frequency: number;
    readonly power: number;
    readonly index: number;
    readonly period: number;
}

export const refinePeak: (
    frequencies: ArrayLike<number>,
    power: ArrayLike<number>,
    index: number
) => RefinedPeak = (
    frequencies: ArrayLike<number>,
    power: ArrayLike<number>,
    index: number
): RefinedPeak => {
    if (index <= 0 || index >= power.length - 1) {
        return {
            frequency: frequencies[index],
            power: power[index],
            index,
            period: 1 / frequencies[index]
        };
    }
    const y0: number = power[index - 1];
    const y1: number = power[index];
    const y2: number = power[index + 1];
    const denominator: number = y0 - 2 * y1 + y2;
    const rawDelta: number = denominator !== 0 ? (0.5 * (y0 - y2)) / denominator : 0;
    const delta: number = Math.abs(rawDelta) < 1 ? rawDelta : 0;
    const df: number = frequencies[1] - frequencies[0];
    const frequency: number = frequencies[index] + delta * df;
    return {
        frequency,
        power: y1 - 0.25 * (y0 - y2) * delta,
        index,
        period: 1 / frequency
    };
};

/**
 * 加窗 DFT 的合成核 —— **修正后的单核**（缺陷 7）。
 *
 *     W_T(θ) = (1/T) Σ_{n<T} w[n]·e^{+i2πnθ}
 *
 * 加窗频谱是 `G[k] = Σₘ cₘ·W_T(m/ρ − k/T)`。`w ≡ 1` 时退化为 Dirichlet 核。
 * **不可**写成「窗变换与 D_T 的乘积」。
 */
export const windowedKernel: (
    window: ArrayLike<number>,
    theta: number
) => { real: number; imaginary: number } = (
    window: ArrayLike<number>,
    theta: number
): { real: number; imaginary: number } => {
    const t: number = window.length;
    let sr = 0;
    let si = 0;
    for (let n = 0; n < t; n += 1) {
        const a: number = 2 * Math.PI * n * theta;
        sr += window[n] * Math.cos(a);
        si += window[n] * Math.sin(a);
    }
    return { real: sr / t, imaginary: si / t };
};

/** Dirichlet 核 `D_T(θ) = (1/T) Σ_{n<T} e^{i2πnθ}`。 */
export const dirichletKernel: (length: number, theta: number) => { real: number; imaginary: number } = (
    length: number,
    theta: number
): { real: number; imaginary: number } => {
    const numerator: number = Math.sin(Math.PI * length * theta);
    const denominator: number = length * Math.sin(Math.PI * theta);
    if (Math.abs(denominator) < 1e-300) {
        return { real: 1, imaginary: 0 };
    }
    const magnitude: number = numerator / denominator;
    const phase: number = Math.PI * (length - 1) * theta;
    return { real: magnitude * Math.cos(phase), imaginary: magnitude * Math.sin(phase) };
};

/** 常用窗函数。 */
export const windows: {
    readonly rect: (length: number) => Float64Array;
    readonly hann: (length: number) => Float64Array;
    readonly hamming: (length: number) => Float64Array;
    readonly blackman: (length: number) => Float64Array;
} = {
    rect: (length: number): Float64Array => Float64Array.from({ length }, () => 1),
    hann: (length: number): Float64Array =>
        Float64Array.from(
            { length },
            (_, n) => 0.5 * (1 - Math.cos((2 * Math.PI * n) / (length - 1)))
        ),
    hamming: (length: number): Float64Array =>
        Float64Array.from(
            { length },
            (_, n) => 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (length - 1))
        ),
    blackman: (length: number): Float64Array =>
        Float64Array.from({ length }, (_, n) => {
            const t: number = (2 * Math.PI * n) / (length - 1);
            return 0.42 - 0.5 * Math.cos(t) + 0.08 * Math.cos(2 * t);
        })
};

/**
 * 找若干彼此分离的峰，最强的在前。
 *
 * `minSeparation` 是新峰与已选峰之间在**周期**上的最小相对距离，
 * 用于避免把同一个宽峰的多个栅格点当成多个峰。
 */
export const findPeaks: (
    frequencies: ArrayLike<number>,
    power: ArrayLike<number>,
    options?: { count?: number; minSeparation?: number }
) => readonly RefinedPeak[] = (
    frequencies: ArrayLike<number>,
    power: ArrayLike<number>,
    options: { count?: number; minSeparation?: number } = {}
): readonly RefinedPeak[] => {
    const count: number = options.count ?? 5;
    const minSeparation: number = options.minSeparation ?? 0.15;
    const candidates: number[] = [];
    // 上界含 Nyquist 那一格：周期为 2 的序列（逐点正负交替）只在 Nyquist 处有能量，
    // 把它排除掉会让这种最纯粹的周期信号变得「看不见」。refinePeak 在端点处不插值，
    // 会原样返回该格，故这里不需要额外处理。
    for (let k = 1; k < power.length; k += 1) {
        const next: number = k + 1 < power.length ? power[k + 1] : -Infinity;
        if (power[k] > power[k - 1] && power[k] >= next) {
            candidates.push(k);
        }
    }
    candidates.sort((a, b) => power[b] - power[a]);

    const chosen: RefinedPeak[] = [];
    for (const k of candidates) {
        if (chosen.length >= count) {
            break;
        }
        const peak: RefinedPeak = refinePeak(frequencies, power, k);
        if (peak.frequency <= 0) {
            continue;
        }
        const tooClose: boolean = chosen.some(
            (c) => Math.abs(c.period - peak.period) / peak.period < minSeparation
        );
        if (!tooClose) {
            chosen.push(peak);
        }
    }
    return chosen;
};

/** 周期图总功率（不含直流）。 */
export const totalPower: (power: ArrayLike<number>) => number = (power: ArrayLike<number>): number => {
    let sum = 0;
    for (let i = 1; i < power.length; i += 1) {
        sum += power[i];
    }
    return sum;
};

/** 频带内功率占比，用于「能量集中在哪些周期」这类问答。 */
export const bandPowerShare: (
    frequencies: ArrayLike<number>,
    power: ArrayLike<number>,
    period: number,
    relativeBand?: number
) => number = (
    frequencies: ArrayLike<number>,
    power: ArrayLike<number>,
    period: number,
    relativeBand: number = 0.1
): number => {
    const total: number = totalPower(power);
    if (total <= 0) {
        return 0;
    }
    const target: number = 1 / period;
    let sum = 0;
    for (let i = 1; i < power.length; i += 1) {
        if (Math.abs(frequencies[i] - target) / target <= relativeBand) {
            sum += power[i];
        }
    }
    return sum / total;
};

export { demean };
