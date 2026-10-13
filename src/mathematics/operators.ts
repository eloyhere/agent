import { Complex, polarAsRadian } from "./complex";
import { Epicycle } from "./epicycle";
import { Waveform } from "./waveform";
import { isNumber } from "../validation/type";

/**
 * 书本的算子，作用在周转圆与波形上。
 *
 * 波形 `W(x) = Σ Aₖ·e^{i(ωₖx + φₖ)}` 的求值就是 CEAF，故书本各算子都有精确的
 * 三项作用 —— 不需要近似，也没有隐藏状态要同步。
 *
 *   Translation   ex:phase_functor_action
 *                 平移 τ：φₖ ← φₖ − ωₖτ
 *                 注意这是**依赖频率**的相位旋转，不是全局相位 —— 这正是位置信息
 *                 在这套表示里是「原生」的原因。
 *   Rotation      乘 e^{iθ}：φₖ ← φₖ + θ（全局相位）
 *   Scaling       Aₖ ← λAₖ
 *   Period        周期性对象自身的周期
 *
 *   Periodic convolution   def:periodic_convolution
 *                 Ω = ω₁ + ω₂，Φ = φ₁ + φ₂，振幅再乘相位相似核
 *   Trend convolution      def:tendency_convolution
 *                 T_mn = A_m B_n e^{i(θ_m − φ_n)} = c_m · conj(d_n)
 *   相似核        prop:twelve_to_six
 *                 Sim⁽¹⁾=cos d、Sim⁽²⁾=sin d、Sim⁽³⁾=cos 2d、Sim⁽⁴⁾=sin 2d、
 *                 Sim⁽⁵⁾=cos(d/2)、Sim⁽⁶⁾=sin(d/2)
 */

/** 对单个周转圆平移 τ。 */
export const translateEpicycle: (epicycle: Epicycle, tau: number) => Epicycle = (
    epicycle: Epicycle,
    tau: number
): Epicycle => {
    if (!isNumber(tau)) {
        throw new TypeError("tau must be a number");
    }
    return new Epicycle(
        epicycle.getAmplitude(),
        epicycle.getOmega(),
        epicycle.getPhase() - epicycle.getOmega() * tau
    );
};

/**
 * 平移整个波形。
 *
 * `W(x − τ)`：每个周转圆的相位按其自身频率旋转，多项式部分按泰勒平移
 * `pₙ(x−τ)ⁿ` 展开 —— 多项式在这里会升次，故平移后仍是本函数类内的对象。
 */
export const translate: (waveform: Waveform, tau: number) => Waveform = (
    waveform: Waveform,
    tau: number
): Waveform => {
    if (!isNumber(tau)) {
        throw new TypeError("tau must be a number");
    }
    // 多项式部分：p(x − τ)，逐次用二项式展开
    const source: readonly Complex[] = waveform.getPolynomial();
    const shifted: Complex[] = new Array<Complex>(source.length).fill(new Complex(0, 0));
    for (let n = 0; n < source.length; n += 1) {
        // (x − τ)ⁿ = Σ_{k≤n} C(n,k) xᵏ (−τ)^{n−k}
        for (let k = 0; k <= n; k += 1) {
            const coefficient: Complex = source[n].multiply(binomial(n, k)).multiply(
                Math.pow(-tau, n - k)
            );
            shifted[k] = shifted[k].add(coefficient);
        }
    }
    return new Waveform({
        epicycles: waveform.getEpicycles().map((e) => translateEpicycle(e, tau)),
        polynomial: shifted
    });
};

/** 全局相位旋转：乘 e^{iθ}。 */
export const phaseRotate: (waveform: Waveform, theta: number) => Waveform = (
    waveform: Waveform,
    theta: number
): Waveform => {
    if (!isNumber(theta)) {
        throw new TypeError("theta must be a number");
    }
    return new Waveform({
        epicycles: waveform.getEpicycles().map(
            (e) => new Epicycle(e.getAmplitude(), e.getOmega(), e.getPhase() + theta)
        ),
        polynomial: waveform.getPolynomial().map((c) => c.multiply(polarAsRadian(theta)))
    });
};

/** 缩放（书本的 Scaling 算子）。 */
export const scale: (waveform: Waveform, factor: number) => Waveform = (
    waveform: Waveform,
    factor: number
): Waveform => waveform.times(factor);

/**
 * 书本的相位函子 `Φ_{θ,T}(f)(t) = e^{iθ}·f(t − θT/2π)`。
 *
 * 平移量与全局相位复合：`φₖ ← φₖ + θ − ωₖ·(θT/2π)`。
 */
export const phaseFunctor: (waveform: Waveform, theta: number, period: number) => Waveform = (
    waveform: Waveform,
    theta: number,
    period: number
): Waveform => phaseRotate(translate(waveform, (theta * period) / (2 * Math.PI)), theta);

/** 二项式系数 C(n, k)，小整数用乘法公式。 */
const binomial: (n: number, k: number) => number = (n: number, k: number): number => {
    if (k < 0 || k > n) {
        return 0;
    }
    let result = 1;
    for (let i = 1; i <= k; i += 1) {
        result = (result * (n - k + i)) / i;
    }
    return result;
};

/** 六个相似核在相位差 d 处的取值，各自分实部（余弦）与虚部（正弦）。 */
export interface SimilarityKernels {
    readonly cosine: number;
    readonly sine: number;
}

/** 取第 d 个相似核（d = 1..6），返回 `{cosine, sine}` 两支。 */
export const similarityKernels: (delta: number, d: number) => SimilarityKernels = (
    delta: number,
    d: number
): SimilarityKernels => {
    if (!isNumber(delta)) {
        throw new TypeError("delta must be a number");
    }
    // 1,3,5 取倍角 1/2/0.5；2,4,6 用同一倍角的正弦支
    const scaleFactor: number = d === 3 || d === 4 ? 2 : d === 5 || d === 6 ? 0.5 : 1;
    return {
        cosine: Math.cos(scaleFactor * delta),
        sine: Math.sin(scaleFactor * delta)
    };
};

/**
 * 两个周转圆之间的标量相似度：第 d 个核在相位差处的取值。
 * d = 1, 3, 5 取余弦核；d = 2, 4, 6 取正弦核。
 */
export const similarity: (a: Epicycle, b: Epicycle, d?: number) => number = (
    a: Epicycle,
    b: Epicycle,
    d: number = 1
): number => {
    if (d < 1 || d > 6 || !Number.isInteger(d)) {
        throw new RangeError(`similarity: d must be an integer 1..6, got ${d}`);
    }
    const kernels: SimilarityKernels = similarityKernels(a.getPhase() - b.getPhase(), d);
    return d % 2 === 1 ? kernels.cosine : kernels.sine;
};

/**
 * 周期卷积（`def:periodic_convolution`）。
 *
 * 每一对周转圆复合：频率**相加**、相位**相加**、振幅相乘（再乘相位相似核，
 * 书本的定义里含这一项）。书本的要点是：这个算子保留**所有**频率对，而不是丢弃。
 *
 * @param d             相似核编号 1..6，默认 1（余弦）
 * @param useSimilarity 传 false 得到纯复合 A₁A₂，即书本所说退化为经典卷积的那一支
 * @param keep          保留的周转圆上限，按 |振幅| 由大到小
 */
export const periodicConvolve: (
    a: Waveform,
    b: Waveform,
    options?: { d?: number; useSimilarity?: boolean; keep?: number }
) => Waveform = (
    a: Waveform,
    b: Waveform,
    options: { d?: number; useSimilarity?: boolean; keep?: number } = {}
): Waveform => {
    const d: number = options.d ?? 1;
    const useSimilarity: boolean = options.useSimilarity ?? true;
    const keep: number = options.keep ?? 64;

    const composed: { amplitude: number; omega: number; phase: number }[] = [];
    for (const x of a.getEpicycles()) {
        for (const y of b.getEpicycles()) {
            const kernel: number = useSimilarity ? similarity(x, y, d) : 1;
            composed.push({
                amplitude: x.getAmplitude() * y.getAmplitude() * kernel,
                omega: x.getOmega() + y.getOmega(),      // 频率复合
                phase: x.getPhase() + y.getPhase()       // 相位复合
            });
        }
    }
    composed.sort((p, q) => Math.abs(q.amplitude) - Math.abs(p.amplitude));

    const epicycles: Epicycle[] = [];
    for (const item of composed.slice(0, keep)) {
        // 相似核可能为负，符号由 Epicycle 的构造函数折进相位
        epicycles.push(new Epicycle(item.amplitude, item.omega, item.phase));
    }
    // 直流分量的复合
    const constantA: Complex = polynomialConstant(a);
    const constantB: Complex = polynomialConstant(b);
    return new Waveform({
        epicycles,
        polynomial: [constantA.multiply(constantB)]
    });
};

/** 多项式部分的常数项（次数 0 的系数），无则返回 0。 */
const polynomialConstant: (waveform: Waveform) => Complex = (waveform: Waveform): Complex => {
    const polynomial: readonly Complex[] = waveform.getPolynomial();
    return polynomial.length > 0 ? polynomial[0] : new Complex(0, 0);
};

/** 趋势卷积（`def:tendency_convolution`）的结果矩阵。 */
export interface TrendMatrix {
    /** 实部，行主序，长度 m×n。 */
    readonly real: Float64Array;
    /** 虚部，行主序。 */
    readonly imaginary: Float64Array;
    readonly rows: number;
    readonly columns: number;
}

/**
 * 趋势卷积：`T_mn = c_m · conj(d_n)` —— 振幅之积配**相位之差**。
 */
export const trendConvolution: (a: Waveform, b: Waveform) => TrendMatrix = (
    a: Waveform,
    b: Waveform
): TrendMatrix => {
    const m: number = a.getEpicycles().length;
    const n: number = b.getEpicycles().length;
    const real = new Float64Array(m * n);
    const imaginary = new Float64Array(m * n);
    for (let i = 0; i < m; i += 1) {
        const p: Epicycle = a.getEpicycles()[i];
        for (let j = 0; j < n; j += 1) {
            const q: Epicycle = b.getEpicycles()[j];
            const delta: number = p.getPhase() - q.getPhase();
            const amplitude: number = p.getAmplitude() * q.getAmplitude();
            real[i * n + j] = amplitude * Math.cos(delta);
            imaginary[i * n + j] = amplitude * Math.sin(delta);
        }
    }
    return { real, imaginary, rows: m, columns: n };
};

/**
 * 趋势卷积的对角线。波形与自身作趋势卷积时，对角项为 `Aₖ²`，实数且非负 ——
 * 这正是书本把自相关读作「趋势卷积在对角线上的投影」的依据
 * （验证报告中的缺陷 9 关乎与之配套的时域求和的归一化）。
 */
export const trendDiagonal: (waveform: Waveform) => readonly number[] = (
    waveform: Waveform
): readonly number[] => waveform.getEpicycles().map((e) => e.getAmplitude() ** 2);
