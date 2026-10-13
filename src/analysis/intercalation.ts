/**
 * 置闰诊断：连分数、最佳逼近与泄漏界。
 *
 * 这是书本「采样 / 置闰」机制的量化核心，也是本智能体回答
 * 「该取多长的窗口」「历法为何是十九年七闰」这类问题的依据。
 */

/** `value` 的连分数展开 `[a₀; a₁, a₂, …]`。 */
export const continuedFraction: (value: number, maxTerms?: number, tolerance?: number) => readonly number[] = (
    value: number,
    maxTerms: number = 20,
    tolerance: number = 1e-14
): readonly number[] => {
    const out: number[] = [];
    let current: number = value;
    for (let i = 0; i < maxTerms; i += 1) {
        const a: number = Math.floor(current);
        out.push(a);
        const fraction: number = current - a;
        if (Math.abs(fraction) < tolerance) {
            break;
        }
        current = 1 / fraction;
        if (!Number.isFinite(current)) {
            break;
        }
    }
    return out;
};

/** 连分数的收敛子 `pₖ/qₖ`。 */
export interface Convergent {
    readonly numerator: number;
    readonly denominator: number;
    readonly index: number;
}

export const convergents: (terms: readonly number[]) => readonly Convergent[] = (
    terms: readonly number[]
): readonly Convergent[] => {
    const out: Convergent[] = [];
    let p0 = 0;
    let p1 = 1;
    let q0 = 1;
    let q1 = 0;
    for (let i = 0; i < terms.length; i += 1) {
        const a: number = terms[i];
        const p: number = a * p1 + p0;
        const q: number = a * q1 + q0;
        out.push({ numerator: p, denominator: q, index: i });
        p0 = p1;
        p1 = p;
        q0 = q1;
        q1 = q;
    }
    return out;
};

/** 到最近整数的距离 `‖x‖_ℤ`。 */
export const distanceToInteger: (value: number) => number = (value: number): number =>
    Math.abs(value - Math.round(value));

/**
 * 书本给出的泄漏界：窗口长 `M`、周期比 `rho` 时
 *
 *     η ≤ (π²/3)·δ_M²,   δ_M = ‖M·rho‖_ℤ
 *
 * 置闰的量化收益：把 `M` 取在连分数分母上，`δ_M ~ 1/q_{k+1}`，
 * 于是 `η` 从 O(1) 掉到 O(M⁻²)。
 */
export const leakBound: (delta: number) => number = (delta: number): number =>
    (Math.PI ** 2 / 3) * delta * delta;

/** 候选窗口长度按泄漏界排序的结果。 */
export interface WindowCandidate {
    readonly length: number;
    readonly delta: number;
    readonly eta: number;
    readonly isConvergentDenominator: boolean;
}

export const rankWindowLengths: (
    rho: number,
    options?: { maxLength?: number; top?: number }
) => { readonly rows: readonly WindowCandidate[]; readonly terms: readonly number[]; readonly convergents: readonly Convergent[] } = (
    rho: number,
    options: { maxLength?: number; top?: number } = {}
): { readonly rows: readonly WindowCandidate[]; readonly terms: readonly number[]; readonly convergents: readonly Convergent[] } => {
    const maxLength: number = options.maxLength ?? 200;
    const top: number = options.top ?? 12;
    const terms: readonly number[] = continuedFraction(rho, 16);
    const convergentList: readonly Convergent[] = convergents(terms);
    const denominators = new Set(convergentList.map((c) => c.denominator));

    const rows: WindowCandidate[] = [];
    for (let length = 1; length <= maxLength; length += 1) {
        const delta: number = distanceToInteger(length * rho);
        rows.push({
            length,
            delta,
            eta: leakBound(delta),
            isConvergentDenominator: denominators.has(length)
        });
    }
    rows.sort((a, b) => a.eta - b.eta);
    return {
        rows: rows.slice(0, top),
        terms,
        convergents: convergentList.filter((c) => c.denominator <= maxLength)
    };
};

/** 平衡置闰：把 `count` 次置闰尽可能均匀地放到 `slots` 个周期里。 */
export const balancedIntercalation: (count: number, slots: number) => readonly number[] = (
    count: number,
    slots: number
): readonly number[] => {
    if (count <= 0 || slots <= 0 || count > slots) {
        throw new RangeError("balancedIntercalation: need 0 < count <= slots");
    }
    const indices: number[] = [];
    for (let i = 1; i <= count; i += 1) {
        indices.push(Math.floor((i * slots) / count) - 1);
    }
    return indices;
};

/** 默冬章式的年—月比分析。 */
export interface MetonicAnalysis {
    readonly rho: number;
    readonly terms: readonly number[];
    readonly convergents: readonly Convergent[];
    readonly years: number;
    readonly months: number;
    readonly intercalations: number;
    readonly delta: number;
    readonly eta: number;
    /** 第七个收敛子的分子（书本验证报告缺陷 13 修正为 4131，原为 3896）。 */
    readonly seventhNumerator: number;
    readonly seventhDenominator: number;
}

export const metonicAnalysis: (
    yearLength: number,
    monthLength: number,
    options?: { years?: number }
) => MetonicAnalysis = (
    yearLength: number,
    monthLength: number,
    options: { years?: number } = {}
): MetonicAnalysis => {
    const years: number = options.years ?? 19;
    const rho: number = yearLength / monthLength;
    const terms: readonly number[] = continuedFraction(rho, 12);
    const convergentList: readonly Convergent[] = convergents(terms);
    const months: number = Math.round(years * rho);
    const delta: number = Math.abs(years * rho - months);
    // 第 7 个收敛子（下标 6）—— 书本原写 3896/315，实为 4131/334
    const seventh: Convergent | undefined = convergentList[6];
    return {
        rho,
        terms,
        convergents: convergentList,
        years,
        months,
        intercalations: months - years * 12,
        delta,
        eta: leakBound(delta),
        seventhNumerator: seventh ? seventh.numerator : 0,
        seventhDenominator: seventh ? seventh.denominator : 1
    };
};
