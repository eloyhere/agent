import { Complex, polarAsRadian } from "../mathematics/complex";
import { Epicycle } from "../mathematics/epicycle";
import { Waveform } from "../mathematics/waveform";
import type { BpeTokenizer } from "./bpe";

/**
 * token → 周转圆组。
 *
 * 书本已有这个对象。`ex:dft_epicycle` 说 DFT 对应有限周转圆组
 *
 *     E_f = { ( |f̂(k)|/T₀ , 2πk/T₀ , arg f̂(k) ) }     k = 0 .. T₀−1
 *
 * 而 `def:eaf_agent` 的 CEAF 就是它求和。于是把 token 的字节当作**周期 L 的一段
 * 周期信号**（**词的周期就是它的长度**），分解为 `(A_k, 2πk/L, φ_k)`：
 *
 *     s(n) = b_n − 128,   n = 0 .. L−1
 *     c_k  = (1/L) Σₙ s(n)·e^{−2πikn/L}
 *
 * **符号谐波**：保留 `k = 0, ±1, ±2, …`，与书本的 `k = 0 .. T₀−1` 一致。
 * 只留 `k ≥ 0` 再靠共轭折叠确实能重建（信号是实的），但**会破坏代数**：
 * Rotation 乘 `e^{iθ}` 使信号变复，卷积使频率相加，两者都跑出半谱之外。
 *
 * 直流分量（k = 0）落在 `Waveform` 的多项式部分，谐波落在周转圆部分 ——
 * 于是 token 直接就是一个 `Waveform`。
 */

const CENTER = 128;

/** 长度为 `L` 的实信号精确重建所需的谐波编号（不含 k = 0）。 */
export const harmonicIndices: (length: number, maxOrder?: number | null) => readonly number[] = (
    length: number,
    maxOrder: number | null = null
): readonly number[] => {
    if (length <= 1) {
        return [];
    }
    const maxK: number = Math.floor(length / 2);
    const out: number[] = [];
    for (let k = 1; k <= maxK; k += 1) {
        out.push(k);
    }
    // 偶数长度时 Nyquist 与其负孪生同频，故略去 −L/2，使总数恰为 L
    const negativeLimit: number = length % 2 === 0 ? maxK - 1 : maxK;
    for (let k = 1; k <= negativeLimit; k += 1) {
        out.push(-k);
    }
    out.sort((a, b) => a - b);
    if (maxOrder === null) {
        return out;
    }
    return out.filter((k) => Math.abs(k) <= maxOrder);
};

/** 把一个字节序列编码为周转圆组（即 `Waveform`）。 */
export const encodeBytes: (
    bytes: Uint8Array,
    options?: { maxOrder?: number | null; center?: number }
) => Waveform = (
    bytes: Uint8Array,
    options: { maxOrder?: number | null; center?: number } = {}
): Waveform => {
    const maxOrder: number | null = options.maxOrder ?? null;
    const center: number = options.center ?? CENTER;
    const length: number = bytes.length;

    if (length === 0) {
        return Waveform.zero();
    }
    if (length === 1) {
        // 只有一个采样：没有谐波，只剩直流
        return Waveform.constant(new Complex(bytes[0] - center, 0));
    }

    const signal = new Float64Array(length);
    for (let n = 0; n < length; n += 1) {
        signal[n] = bytes[n] - center;
    }

    const indices: readonly number[] = harmonicIndices(length, maxOrder);
    const epicycles: Epicycle[] = [];
    for (const k of indices) {
        let sr = 0;
        let si = 0;
        for (let n = 0; n < length; n += 1) {
            const a = (-2 * Math.PI * k * n) / length;
            sr += signal[n] * Math.cos(a);
            si += signal[n] * Math.sin(a);
        }
        const coefficient = new Complex(sr / length, si / length);
        const amplitude: number = coefficient.magnitude();
        if (amplitude < 1e-15) {
            continue; // 该谐波为零，不必留下一个振幅为 0 的周转圆
        }
        epicycles.push(new Epicycle(amplitude, (2 * Math.PI * k) / length, coefficient.argument()));
    }

    let dc = 0;
    for (let n = 0; n < length; n += 1) {
        dc += signal[n];
    }
    dc /= length;
    return new Waveform({ epicycles, polynomial: [new Complex(dc, 0), new Complex(0, 0), new Complex(0, 0)] });
};

/** 一个 token 及其周转圆组。 */
export interface EncodedToken {
    readonly position: number;
    readonly id: number;
    readonly text: string;
    readonly bytes: Uint8Array;
    readonly waveform: Waveform;
    /** 词的周期就是它的字节长度。 */
    readonly period: number;
}

/** 把文本切成 token 并逐个编码为周转圆组。 */
export const tokenizeToWaveforms: (
    text: string,
    tokenizer: BpeTokenizer,
    options?: { maxOrder?: number | null }
) => readonly EncodedToken[] = (
    text: string,
    tokenizer: BpeTokenizer,
    options: { maxOrder?: number | null } = {}
): readonly EncodedToken[] => {
    const ids: number[] = tokenizer.encode(text);
    const out: EncodedToken[] = [];
    for (let position = 0; position < ids.length; position += 1) {
        const id = ids[position];
        const bytes = tokenizer.tokenBytes(id);
        out.push({
            position,
            id,
            text: tokenizer.tokenText(id),
            bytes,
            waveform: encodeBytes(bytes, options),
            period: bytes.length
        });
    }
    return out;
};

/**
 * 重建字节序列，用于校验编码是否无损。
 * 用 `Waveform.value(n)` 在整数采样点上取值即可 —— 这正是 CEAF 本身。
 */
export const reconstructBytes: (waveform: Waveform, length: number, center?: number) => Uint8Array = (
    waveform: Waveform,
    length: number,
    center: number = CENTER
): Uint8Array => {
    const out = new Uint8Array(length);
    for (let n = 0; n < length; n += 1) {
        const value = waveform.value(n).getReal() + center;
        out[n] = Math.max(0, Math.min(255, Math.round(value)));
    }
    return out;
};

/** 极坐标复数，供上层构造使用。 */
export { polarAsRadian };
