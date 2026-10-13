/**
 * 快速傅里叶变换。
 *
 * 约定：正变换 `X_k = Σₙ xₙ·e^{-2πi nk/N}`，与书本归一化 DFT
 * `f̂(k) = (1/T)Σₙ f(n)e^{-2πi nk/T}` 只差调用方自己补的 `1/T`。
 *
 * 两个例程：
 *   `fftRadix2`  长度必须为 2 的幂，原地迭代 Cooley–Tukey
 *   `dft`        任意长度；非 2 的幂时走 Bluestein（chirp-z），复杂度仍是 O(M log M)
 *
 * 之所以要支持任意长度：书本数值验证里那个 `N = 3312` 的太阳黑子序列既不是
 * 2 的幂，也不是 3 或 5 的幂，补零会改变频率栅格、破坏那个已验证的 `132.48`
 * 峰值位置。所以必须在对齐的长度上做变换。
 */

/** 原地基 2 快速傅里叶变换。`real` 与 `imaginary` 长度须相同且为 2 的幂。 */
export const fftRadix2: (real: Float64Array, imaginary: Float64Array, inverse?: boolean) => void = (
    real: Float64Array,
    imaginary: Float64Array,
    inverse: boolean = false
): void => {
    const n: number = real.length;
    if (n <= 1) {
        return;
    }
    if ((n & (n - 1)) !== 0) {
        throw new RangeError(`fftRadix2: length ${n} is not a power of two`);
    }

    // 位反转置换
    for (let i = 1, j = 0; i < n; i += 1) {
        let bit: number = n >> 1;
        for (; (j & bit) !== 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            const tr: number = real[i];
            real[i] = real[j];
            real[j] = tr;
            const ti: number = imaginary[i];
            imaginary[i] = imaginary[j];
            imaginary[j] = ti;
        }
    }

    for (let length = 2; length <= n; length <<= 1) {
        const half: number = length >> 1;
        const angle: number = ((inverse ? 2 : -2) * Math.PI) / length;
        const wr: number = Math.cos(angle);
        const wi: number = Math.sin(angle);
        for (let i = 0; i < n; i += length) {
            let cr = 1;
            let ci = 0;
            for (let k = 0; k < half; k += 1) {
                const a: number = i + k;
                const b: number = a + half;
                const vr: number = real[b] * cr - imaginary[b] * ci;
                const vi: number = real[b] * ci + imaginary[b] * cr;
                real[b] = real[a] - vr;
                imaginary[b] = imaginary[a] - vi;
                real[a] += vr;
                imaginary[a] += vi;
                const nextCr: number = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nextCr;
            }
        }
    }

    if (inverse) {
        for (let i = 0; i < n; i += 1) {
            real[i] /= n;
            imaginary[i] /= n;
        }
    }
};

/** 不小于 `value` 的最小 2 的幂。 */
const nextPowerOfTwo: (value: number) => number = (value: number): number => {
    let n = 1;
    while (n < value) {
        n <<= 1;
    }
    return n;
};

/**
 * 任意长度的离散傅里叶变换。
 *
 * Bluestein 算法：由 `nk = (n² + k² − (k−n)²)/2` 得
 *
 *     X_k = e^{-iπk²/N} · Σₙ [ xₙ·e^{-iπn²/N} ] · e^{+iπ(k−n)²/N}
 *
 * 即一次卷积，用基 2 快速傅里叶变换求值。
 */
export const dft: (
    real: Float64Array,
    imaginary: Float64Array,
    inverse?: boolean
) => { real: Float64Array; imaginary: Float64Array } = (
    real: Float64Array,
    imaginary: Float64Array,
    inverse: boolean = false
): { real: Float64Array; imaginary: Float64Array } => {
    const n: number = real.length;
    if (n === 0) {
        return { real: new Float64Array(0), imaginary: new Float64Array(0) };
    }
    if ((n & (n - 1)) === 0) {
        const r: Float64Array = Float64Array.from(real);
        const i2: Float64Array = Float64Array.from(imaginary);
        fftRadix2(r, i2, inverse);
        return { real: r, imaginary: i2 };
    }

    // 正变换用 e^{-2πink/N}；逆变换共轭核
    const sign: number = inverse ? 1 : -1;
    const modTwoN: number = 2 * n;
    const phase = (m: number): number => {
        const t: number = ((m % modTwoN) + modTwoN) % modTwoN; // 压小指数，避免大数失真
        return (sign * Math.PI * ((t * t) % modTwoN)) / n;
    };

    const m: number = nextPowerOfTwo(2 * n - 1);
    const ar = new Float64Array(m);
    const ai = new Float64Array(m);
    const br = new Float64Array(m);
    const bi = new Float64Array(m);

    for (let k = 0; k < n; k += 1) {
        const p: number = phase(k);
        const c: number = Math.cos(p);
        const s: number = Math.sin(p);
        ar[k] = real[k] * c - imaginary[k] * s;
        ai[k] = real[k] * s + imaginary[k] * c;
    }
    for (let k = 0; k < n; k += 1) {
        const p: number = -phase(k); // b 是共轭 chirp
        br[k] = Math.cos(p);
        bi[k] = Math.sin(p);
        if (k > 0) {
            br[m - k] = br[k];
            bi[m - k] = bi[k];
        }
    }

    fftRadix2(ar, ai, false);
    fftRadix2(br, bi, false);
    for (let k = 0; k < m; k += 1) {
        const tr: number = ar[k] * br[k] - ai[k] * bi[k];
        const ti: number = ar[k] * bi[k] + ai[k] * br[k];
        ar[k] = tr;
        ai[k] = ti;
    }
    fftRadix2(ar, ai, true);

    const outReal = new Float64Array(n);
    const outImaginary = new Float64Array(n);
    for (let k = 0; k < n; k += 1) {
        const p: number = phase(k);
        const c: number = Math.cos(p);
        const s: number = Math.sin(p);
        outReal[k] = ar[k] * c - ai[k] * s;
        outImaginary[k] = ar[k] * s + ai[k] * c;
    }
    return { real: outReal, imaginary: outImaginary };
};

/** 实信号的 DFT，返回非冗余的一半 `0 .. ⌊N/2⌋`。 */
export const realDft: (samples: ArrayLike<number>) => { real: Float64Array; imaginary: Float64Array } = (
    samples: ArrayLike<number>
): { real: Float64Array; imaginary: Float64Array } => {
    const n: number = samples.length;
    const input = Float64Array.from(samples);
    const result = dft(input, new Float64Array(n), false);
    const half: number = Math.floor(n / 2) + 1;
    return {
        real: result.real.slice(0, half),
        imaginary: result.imaginary.slice(0, half)
    };
};
