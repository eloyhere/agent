import { dft } from "../mathematics/fft";

/**
 * 周期估计：自相关、R/S Hurst 指数、循环自相关。
 *
 * **修正过的公式（验证报告缺陷 9）**
 *
 * 自相关带 `1/T` 归一化：
 *
 *     R_z(h) = (1/T) Σₜ zₜ · conj(z_{t+h})
 *
 * 这样 `R_z(h) = Σₙ 𝕋ₙₙ·e^{-iωₙh}` 才成立（𝕋ₙₙ 是趋势卷积的对角线）。
 * 书本原来把 `R_z` 定义为**未归一化**的求和，却仍与那个谱形式划等号，
 * 于是差了整整一个周期因子 `T`；两版都已修正。这里用的是修正后的版本。
 */

export const mean: (values: ArrayLike<number>) => number = (values: ArrayLike<number>): number => {
    let sum = 0;
    for (let i = 0; i < values.length; i += 1) {
        sum += values[i];
    }
    return sum / values.length;
};

export const variance: (values: ArrayLike<number>, ddof?: number) => number = (
    values: ArrayLike<number>,
    ddof: number = 1
): number => {
    const m: number = mean(values);
    let sum = 0;
    for (let i = 0; i < values.length; i += 1) {
        const d: number = values[i] - m;
        sum += d * d;
    }
    return sum / (values.length - ddof);
};

/** 去均值。 */
export const demean: (values: ArrayLike<number>) => Float64Array = (
    values: ArrayLike<number>
): Float64Array => {
    const m: number = mean(values);
    const out = new Float64Array(values.length);
    for (let i = 0; i < values.length; i += 1) {
        out[i] = values[i] - m;
    }
    return out;
};

/**
 * 循环自协方差，滞后 `0 .. maxLag`，用 `1/T` 约定：
 *
 *     acov[h] = (1/T) Σ_{t=0}^{T-1} xc[t]·xc[(t+h) mod T]
 *
 * 这正是书本的周期求和，故 Wiener–Khinchin 恒等式成立。
 */
export const autocovariance: (centered: ArrayLike<number>, maxLag?: number | null) => Float64Array = (
    centered: ArrayLike<number>,
    maxLag: number | null = null
): Float64Array => {
    const t: number = centered.length;
    const lagMax: number = maxLag === null ? Math.floor(t / 2) : Math.min(maxLag, t - 1);
    const input = Float64Array.from(centered);
    const transformed = dft(input, new Float64Array(t), false);
    const powerReal = new Float64Array(t);
    const powerImaginary = new Float64Array(t);
    for (let k = 0; k < t; k += 1) {
        powerReal[k] =
            transformed.real[k] * transformed.real[k] +
            transformed.imaginary[k] * transformed.imaginary[k];
    }
    const correlation = dft(powerReal, powerImaginary, true);
    const out = new Float64Array(lagMax + 1);
    for (let h = 0; h <= lagMax; h += 1) {
        out[h] = correlation.real[h] / t;
    }
    return out;
};

/** 滞后 0 处归一化为 1 的自相关。 */
export const autocorrelation: (
    values: ArrayLike<number>,
    options?: { maxLag?: number | null; detrend?: boolean }
) => Float64Array = (
    values: ArrayLike<number>,
    options: { maxLag?: number | null; detrend?: boolean } = {}
): Float64Array => {
    const maxLag: number | null = options.maxLag ?? null;
    const detrend: boolean = options.detrend ?? true;
    const working: Float64Array = detrend ? demean(values) : Float64Array.from(values);
    const acov: Float64Array = autocovariance(working, maxLag);
    const a0: number = acov[0];
    const out = new Float64Array(acov.length);
    for (let i = 0; i < acov.length; i += 1) {
        out[i] = a0 !== 0 ? acov[i] / a0 : 0;
    }
    return out;
};

/**
 * 经典重标极差（R/S）Hurst 指数。
 *
 * `H > 0.5` 表示长程正持续；`H < 0.5` 反持续。注意 R/S 对短序列与趋势有上偏，
 * 故它只能作为**指示**，不能当作结论。
 */
export const hurstRS: (
    values: ArrayLike<number>,
    options?: { minChunk?: number; factor?: number }
) => number = (
    values: ArrayLike<number>,
    options: { minChunk?: number; factor?: number } = {}
): number => {
    const minChunk: number = options.minChunk ?? 16;
    const factor: number = options.factor ?? 1.6;
    const n: number = values.length;
    const centered: Float64Array = demean(values);

    const sizes: number[] = [];
    for (let s = minChunk; s <= Math.floor(n / 4); s = Math.floor(s * factor)) {
        sizes.push(s);
    }

    const xs: number[] = [];
    const ys: number[] = [];
    for (const size of sizes) {
        const ratios: number[] = [];
        for (let start = 0; start + size <= n; start += size) {
            let mu = 0;
            for (let i = start; i < start + size; i += 1) {
                mu += centered[i];
            }
            mu /= size;
            let acc = 0;
            let lo = Infinity;
            let hi = -Infinity;
            let sumSquares = 0;
            for (let i = start; i < start + size; i += 1) {
                const d: number = centered[i] - mu;
                acc += d;
                if (acc < lo) {
                    lo = acc;
                }
                if (acc > hi) {
                    hi = acc;
                }
                sumSquares += d * d;
            }
            const s: number = Math.sqrt(sumSquares / (size - 1));
            if (s > 1e-12) {
                ratios.push((hi - lo) / s);
            }
        }
        if (ratios.length > 0) {
            xs.push(Math.log(size));
            ys.push(Math.log(ratios.reduce((a, b) => a + b, 0) / ratios.length));
        }
    }
    if (xs.length < 2) {
        return Number.NaN;
    }
    const mx: number = mean(xs);
    const my: number = mean(ys);
    let numerator = 0;
    let denominator = 0;
    for (let i = 0; i < xs.length; i += 1) {
        numerator += (xs[i] - mx) * (ys[i] - my);
        denominator += (xs[i] - mx) ** 2;
    }
    return denominator !== 0 ? numerator / denominator : Number.NaN;
};

/** 循环自相关扫描结果。 */
export interface CyclicScan {
    readonly alphas: Float64Array;
    readonly magnitude: Float64Array;
    readonly peakAlpha: number;
    readonly peakPeriod: number;
}

/**
 * 滞后 0 处的循环自相关，作为循环频率 α 的函数：
 *
 *     R^α(0) = (1/N) Σₜ (xₜ − m)²·e^{-2πiαt}
 *
 * 周期为 T 的分量会让它在 α = 1/T 处变大。
 *
 * 平方序列必须**再去一次均值**：否则 α→0 处会竖起一根巨大的直流尖峰，
 * 把真正要找的周期整个盖住。
 */
export const cyclicAutocorrelation: (
    values: ArrayLike<number>,
    options?: { alphaMax?: number; count?: number; detrend?: boolean }
) => CyclicScan = (
    values: ArrayLike<number>,
    options: { alphaMax?: number; count?: number; detrend?: boolean } = {}
): CyclicScan => {
    const alphaMax: number = options.alphaMax ?? 0.25;
    const count: number = options.count ?? 2000;
    const detrend: boolean = options.detrend ?? true;
    const n: number = values.length;
    const working: Float64Array = detrend ? demean(values) : Float64Array.from(values);

    const squared = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
        squared[i] = working[i] * working[i];
    }
    const squaredMean: number = mean(squared);
    for (let i = 0; i < n; i += 1) {
        squared[i] -= squaredMean;
    }

    const alphas = new Float64Array(count);
    const magnitude = new Float64Array(count);
    for (let j = 0; j < count; j += 1) {
        const alpha: number = ((j + 1) / count) * alphaMax;
        alphas[j] = alpha;
        let sr = 0;
        let si = 0;
        for (let t = 0; t < n; t += 1) {
            const p: number = -2 * Math.PI * alpha * t;
            sr += squared[t] * Math.cos(p);
            si += squared[t] * Math.sin(p);
        }
        magnitude[j] = Math.hypot(sr, si) / n;
    }

    let best = 0;
    for (let i = 1; i < magnitude.length; i += 1) {
        if (magnitude[i] > magnitude[best]) {
            best = i;
        }
    }
    return { alphas, magnitude, peakAlpha: alphas[best], peakPeriod: 1 / alphas[best] };
};

/** 自相关的局部极大值，按强度降序。 */
export interface AutocorrelationPeak {
    readonly lag: number;
    readonly value: number;
}

export const autocorrelationPeaks: (
    autocorrelationValues: ArrayLike<number>,
    options?: { minLag?: number; count?: number }
) => readonly AutocorrelationPeak[] = (
    autocorrelationValues: ArrayLike<number>,
    options: { minLag?: number; count?: number } = {}
): readonly AutocorrelationPeak[] => {
    const minLag: number = options.minLag ?? 2;
    const count: number = options.count ?? 5;
    const ac: ArrayLike<number> = autocorrelationValues;
    const peaks: AutocorrelationPeak[] = [];
    for (let h = minLag; h < ac.length - 1; h += 1) {
        if (ac[h] > ac[h - 1] && ac[h] >= ac[h + 1]) {
            peaks.push({ lag: h, value: ac[h] });
        }
    }
    peaks.sort((a, b) => b.value - a.value);
    return peaks.slice(0, count);
};
