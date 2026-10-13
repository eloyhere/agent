import { Complex } from "./complex";
import { Epicycle, isEpicycle } from "./epicycle";
import { isNumber, isTarget } from "../validation/type";

const Identity: Record<string, symbol> = Object.freeze({
    Waveform: Symbol.for("Waveform")
});

/** 是否为本模块的波形实例。 */
export const isWaveform: (target: unknown) => target is Waveform = (
    target: unknown
): target is Waveform => isTarget(target, Identity.Waveform);

/** 构造参数。 */
export interface WaveformInit {
    /** 周转圆分量，每个的 ω 均非零。 */
    readonly epicycles?: readonly Epicycle[];
    /** 多项式部分的系数，下标即次数：`polynomial[n]` 是 xⁿ 的系数。 */
    readonly polynomial?: readonly Complex[];
}

/**
 * 波形：周转圆之和，外加一个多项式部分
 *
 *     W(x) = Σₙ pₙ·xⁿ  +  Σₖ Aₖ·e^{i(ωₖx + φₖ)}
 *
 * 为什么必须带**多项式部分**：常数项的积分是线性项，线性项的积分是二次项 ——
 * 它们都不是周转圆。若只保留周转圆，`integral()` 就无法满足「返回新的复变函数」
 * 这一要求（常数的积分跑出了这个函数类）。补上多项式部分后，求导与积分
 * **双向完全封闭**，两个方法都不必抛错。
 *
 * 两条伴随恒等式（书本 `ch13_adjunction_numeric_example` 的正是第二条）：
 *
 *     W.integral().derivate()   === W          （D∘I = id）
 *     W.derivate().integral()   === W − W(0)   （I∘D = id − 在 0 处取值）
 *
 * 第二条里丢掉的那个常数正是积分常数：`integral()` 取**积分常数为 0 的原函数**
 * （即 `∫₀ˣ` 的那个原函数），故 `derivate()` 抹掉的常数项无法复原 —— 这不是缺陷，
 * 而是微积分基本定理的准确表述。
 *
 * 设计要点：
 *   1. **不可变**：所有运算返回新实例。
 *   2. **系数规范化**：多项式部分去掉尾部多余的零，故零多项式的系数数组为空。
 *      这使 `equals` 成为真正的结构判等。
 *   3. 相位一律弧度，振幅一律非负（由 `Epicycle` 保证）。
 */
export class Waveform {
    /** 周转圆分量。 */
    protected epicycles: Epicycle[];
    /** 多项式系数，下标即次数，尾部无多余的零。 */
    protected polynomial: Complex[];

    declare readonly [Identity.Waveform]: symbol;

    public constructor(init: WaveformInit = {}) {
        // 不对 init 做 isObject 收窄：那会把类型缩成 Record<PropertyKey, unknown>，
        // 后面的 init.epicycles 就取不到 WaveformInit 的字段类型了。
        // 形状由类型系统保证，这里只做元素级的运行期校验。
        const epicycles: readonly Epicycle[] = init.epicycles ?? [];
        const polynomial: readonly Complex[] = init.polynomial ?? [];
        for (const e of epicycles) {
            if (!isEpicycle(e)) {
                throw new TypeError("Every epicycle must be an Epicycle");
            }
        }
        for (const c of polynomial) {
            if (!(c instanceof Complex)) {
                throw new TypeError("Every polynomial coefficient must be a Complex");
            }
        }
        this.epicycles = [...epicycles];
        this.polynomial = Waveform.trim(polynomial);
        Object.defineProperty(this, Identity.Waveform, {
            value: Identity.Waveform,
            enumerable: false,
            writable: false,
            configurable: false
        });
    }

    /** 去掉尾部多余的零系数。 */
    private static trim(coefficients: readonly Complex[]): Complex[] {
        let end: number = coefficients.length;
        while (end > 0 && coefficients[end - 1].magnitude() === 0) {
            end -= 1;
        }
        return coefficients.slice(0, end).map((c) => new Complex(c.getReal(), c.getImaginary()));
    }

    /** 零波形。 */
    public static zero(): Waveform {
        return new Waveform();
    }

    /** 复常数 c。 */
    public static constant(c: Complex): Waveform {
        return new Waveform({ polynomial: [c] });
    }

    /** 只由周转圆构成。 */
    public static fromEpicycles(epicycles: readonly Epicycle[]): Waveform {
        return new Waveform({ epicycles });
    }

    public getEpicycles(): readonly Epicycle[] {
        return this.epicycles;
    }

    public getPolynomial(): readonly Complex[] {
        return this.polynomial;
    }

    /** 多项式的次数；零多项式约定为 −1。 */
    public degree(): number {
        return this.polynomial.length - 1;
    }

    /** 在 `x` 处的取值。 */
    public value(x: number): Complex {
        if (!isNumber(x)) {
            throw new TypeError("Argument must be a number");
        }
        let total: Complex = new Complex(0, 0);
        // 多项式部分用 Horner 法则，次数通常极低（求导/积分各只改变一次次数）
        for (let n = this.polynomial.length - 1; n >= 0; n -= 1) {
            total = total.multiply(x).add(this.polynomial[n]);
        }
        for (const e of this.epicycles) {
            total = total.add(e.value(x));
        }
        return total;
    }

    /**
     * 求导。无参返回新波形，带参返回数值。
     *
     * 逐项作用：多项式部分 `Σ n·pₙ·xⁿ⁻¹`，周转圆部分各自求导
     * （振幅乘 |ω|、相位 ±π/2，见 `Epicycle.derivate`）。结果仍在本函数类内。
     */
    public derivate(): Waveform;
    public derivate(x: number): Complex;
    public derivate(x?: number): Waveform | Complex {
        const scaled: Complex[] = [];
        for (let n = 1; n < this.polynomial.length; n += 1) {
            scaled.push(this.polynomial[n].multiply(n));
        }
        const derivative: Waveform = new Waveform({
            epicycles: this.epicycles.map((e) => e.derivate()),
            polynomial: scaled
        });
        return x === undefined ? derivative : derivative.value(x);
    }

    /**
     * 积分。无参返回新波形，带参返回定积分数值。
     *
     * 逐项作用：多项式部分 `Σ pₙ·xⁿ⁺¹/(n+1)`（次数升一，正是这一点让函数类封闭），
     * 周转圆部分各自积分。
     *
     * 无参版本返回**满足 F(0) = 0 的原函数**，即书本 `ch13_adjunction_numeric_example`
     * 里的 `I` 算子 `∫₀ˣ`。这一归一化必须显式做：逐项原函数在 0 处并不都取 0
     * （周转圆的原函数在 0 处是 `E(0)/(iω)`，多项式部分却因约定而恰好取 0），
     * 若不统一，`I∘D` 就得不到 `W − W(0)`，而会多留一个常数项。
     * 补齐这个常数的正是本类的多项式部分。
     *
     * 带参版本给出 `∫ₐᵇ W`，积分常数自动消去。
     */
    public integral(): Waveform;
    public integral(start: number, end: number): Complex;
    public integral(start?: number, end?: number): Waveform | Complex {
        // 系数 0 的常数项显式占位，使 pₙ 升到 xⁿ⁺¹ 的位置
        const antiderivativeCoefficients: Complex[] = [new Complex(0, 0)];
        for (let n = 0; n < this.polynomial.length; n += 1) {
            antiderivativeCoefficients.push(this.polynomial[n].divide(n + 1));
        }
        const raw: Waveform = new Waveform({
            epicycles: this.epicycles.map((e) => e.integral()),
            polynomial: antiderivativeCoefficients
        });
        // 归一化到 F(0) = 0
        const antiderivative: Waveform = raw.plus(Waveform.constant(raw.value(0).negate()));
        if (start === undefined) {
            return antiderivative;
        }
        if (end === undefined) {
            throw new TypeError("integral(start, end) requires both bounds");
        }
        if (!isNumber(start) || !isNumber(end)) {
            throw new TypeError("Integration bounds must be numbers");
        }
        return antiderivative.value(end).subtract(antiderivative.value(start));
    }

    /** 逐分量相加，构成新的波形。 */
    public plus(other: Waveform): Waveform {
        if (!isWaveform(other)) {
            throw new TypeError("Operand must be a Waveform");
        }
        const length: number = Math.max(this.polynomial.length, other.polynomial.length);
        const polynomial: Complex[] = [];
        for (let n = 0; n < length; n += 1) {
            const a: Complex = n < this.polynomial.length ? this.polynomial[n] : new Complex(0, 0);
            const b: Complex = n < other.polynomial.length ? other.polynomial[n] : new Complex(0, 0);
            polynomial.push(a.add(b));
        }
        return new Waveform({
            epicycles: [...this.epicycles, ...other.epicycles],
            polynomial
        });
    }

    /** 数乘（复数或实数）。 */
    public times(factor: number): Waveform;
    public times(factor: Complex): Waveform;
    public times(factor: number | Complex): Waveform {
        const f: Complex = isNumber(factor) ? new Complex(factor, 0) : factor;
        if (!(f instanceof Complex)) {
            throw new TypeError("Factor must be a number or a Complex");
        }
        const scaledEpicycles: Epicycle[] = [];
        for (const e of this.epicycles) {
            // 缩放振幅只能改模，改方向要落在相位上：把 c·e^{i(ωx+φ)} 读作
            // |c|·A · e^{i(ωx + φ + arg c)}
            const magnitudeScale: number = f.magnitude();
            const phaseShift: number = f.argument();
            scaledEpicycles.push(
                new Epicycle(e.getAmplitude() * magnitudeScale, e.getOmega(), e.getPhase() + phaseShift)
            );
        }
        return new Waveform({
            epicycles: scaledEpicycles,
            polynomial: this.polynomial.map((c) => c.multiply(f))
        });
    }

    /** 结构判等：多项式系数与周转圆列表逐项严格相等。 */
    public equals(other: Waveform): boolean {
        if (!isWaveform(other)) {
            return false;
        }
        if (this.polynomial.length !== other.polynomial.length) {
            return false;
        }
        for (let n = 0; n < this.polynomial.length; n += 1) {
            if (!this.polynomial[n].equals(other.polynomial[n])) {
                return false;
            }
        }
        if (this.epicycles.length !== other.epicycles.length) {
            return false;
        }
        for (let k = 0; k < this.epicycles.length; k += 1) {
            if (!this.epicycles[k].equals(other.epicycles[k])) {
                return false;
            }
        }
        return true;
    }

    /** 形如 `[2 个周转圆 + 1 次多项式]`。 */
    public toString(): string {
        const degree: number = this.degree();
        const poly: string = degree < 0 ? "0" : `degree ${degree}`;
        return `Waveform(${this.epicycles.length} epicycles + ${poly})`;
    }
}
