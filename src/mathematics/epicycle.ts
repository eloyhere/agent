import { Complex, polarAsRadian, isComplex } from "./complex";
import { isNumber, isTarget } from "../validation/type";

const Identity: Record<string, symbol> = Object.freeze({
    Epicycle: Symbol.for("Epicycle")
});

/** 是否为本模块的周转圆实例。 */
export const isEpicycle: (target: unknown) => target is Epicycle = (
    target: unknown
): target is Epicycle => isTarget(target, Identity.Epicycle);

/**
 * 周转圆：一个复指数分量
 *
 *     E(x) = A · e^{i(ωx + φ)}
 *
 * 三要素与书本 `def:eaf_agent` 的 CEAF 参数一一对应，也是
 * `ex:dft_epicycle` 中 `(|f̂(k)|, 2πk/T₀, arg f̂(k))` 三元组的对象化。
 *
 * 设计要点：
 *   1. **不可变**：所有运算返回新实例，自身永不改变，故可安全共享与缓存。
 *      这与 `Complex` 的约定一致。
 *   2. **振幅恒为非负**（`A ≥ 0`），符号一律并入相位。这样振幅就是物理意义上的
 *      「模」，而相位承担全部符号信息；运算中若算出负振幅，会自动折算成
 *      `|A|` 与 `φ + π`，调用方无需关心。
 *   3. **角频率 `ω` 不得为零**：`ω = 0` 时本对象退化为常数，而常数的积分是
 *      线性项、不再是周转圆，`integrate()` 将无法返回同类。常数量属于
 *      `Waveform` 的多项式部分（见 `waveform.ts`），在此显式拒绝，以免留下
 *      一个会抛错的角落。
 *   4. **相位以弧度计**，与 `Complex.argument()` 及各算子统一。
 *
 * 微积分（详见 `differentiate` / `integrate` 的注释）：
 *
 *     E'(x)  = iω · E(x)        —— 仍是同一个 ω 的周转圆
 *     ∫E dx  = E(x) / (iω)      —— 同上
 *
 * 即求导与积分都只在 (振幅, 相位) 上作用，频率保持不变。
 */
export class Epicycle {
    /** 振幅 A ≥ 0。 */
    protected amplitude: number;
    /** 角频率 ω ≠ 0，单位：弧度 / 单位自变量。 */
    protected omega: number;
    /** 初相 φ，单位：弧度。 */
    protected phase: number;

    /**
     * 类型标记，供 `isTarget` 在运行期识别本类实例。
     *
     * 与 `Complex` 同理，必须声明为可被 Reflect 探测的普通属性：私有字段是内部槽，
     * `Reflect.has` 看不见，用它会让 `isEpicycle` 恒为 false。
     */
    declare readonly [Identity.Epicycle]: symbol;

    /**
     * @param amplitude 振幅，取绝对值后存储（符号并入相位）
     * @param omega     角频率，**不得为 0**
     * @param phase     初相（弧度）
     */
    public constructor(amplitude: number = 1, omega: number = 1, phase: number = 0) {
        if (!isNumber(amplitude) || !isNumber(omega) || !isNumber(phase)) {
            throw new TypeError("Amplitude, omega and phase must be numbers");
        }
        if (omega === 0) {
            throw new RangeError(
                "Epicycle requires a non-zero omega: a zero-frequency term is a constant, which belongs to Waveform's polynomial part"
            );
        }
        // 归一化：振幅取模，符号折算进相位
        if (amplitude < 0) {
            this.amplitude = -amplitude;
            this.phase = phase + Math.PI;
        } else {
            this.amplitude = amplitude;
            this.phase = phase;
        }
        this.omega = omega;
        Object.defineProperty(this, Identity.Epicycle, {
            value: Identity.Epicycle,
            enumerable: false,
            writable: false,
            configurable: false
        });
    }

    public getAmplitude(): number {
        return this.amplitude;
    }

    public getOmega(): number {
        return this.omega;
    }

    public getPhase(): number {
        return this.phase;
    }

    /** 该周转圆支撑一个完整周期所需的长度 `2π / |ω|`（可为小数）。 */
    public period(): number {
        return (2 * Math.PI) / Math.abs(this.omega);
    }

    /** 在 `x` 处的取值 E(x) = A·e^{i(ωx + φ)}。 */
    public value(x: number): Complex {
        if (!isNumber(x)) {
            throw new TypeError("Argument must be a number");
        }
        return polarAsRadian(this.omega * x + this.phase, this.amplitude);
    }

    /**
     * 求导。
     *
     * 无参版本返回新函数；带参版本返回数值，二者共用同一段推导：
     *
     *     E'(x) = iω · A·e^{i(ωx + φ)} = |ω|A · e^{i(ωx + φ + sgn(ω)·π/2)}
     *
     * 因为 `i·sgn(ω) = e^{i·sgn(ω)·π/2}`：`ω > 0` 时 `iω = |ω|e^{iπ/2}`，
     * `ω < 0` 时 `iω = |ω|e^{-iπ/2}`。式中的 `|ω|` 保证振幅仍非负，
     * 方向信息全部由相位 +\-(π/2) 承担。
     */
    public derivate(): Epicycle;
    public derivate(x: number): Complex;
    public derivate(x?: number): Epicycle | Complex {
        const derivative: Epicycle = new Epicycle(
            Math.abs(this.omega) * this.amplitude,
            this.omega,
            this.phase + (Math.PI / 2) * Math.sign(this.omega)
        );
        return x === undefined ? derivative : derivative.value(x);
    }

    /**
     * 积分（以 0 为起点的原函数）。
     *
     *     ∫₀ˣ E(t) dt = E(x) / (iω) = (A / |ω|) · e^{i(ωx + φ - sgn(ω)·π/2)}
     *
     * 因为 `1/i = -i = e^{-iπ/2}`，再带上 `1/ω` 的符号。
     *
     * 无参版本返回新函数（**积分常数取 0**，即 `∫₀ˣ`；需要别的常数请自行叠加
     * 一个常数量，那属于 `Waveform` 的多项式部分）；带参版本返回定积分数值，
     * 此时常数自动消去。
     */
    public integral(): Epicycle;
    public integral(start: number, end: number): Complex;
    public integral(start?: number, end?: number): Epicycle | Complex {
        const antiderivative: Epicycle = new Epicycle(
            this.amplitude / Math.abs(this.omega),
            this.omega,
            this.phase - (Math.PI / 2) * Math.sign(this.omega)
        );
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

    /** 按 (振幅, ω, 相位) 三分量判等（严格相等，不做容差）。 */
    public equals(other: Epicycle): boolean {
        return (
            isEpicycle(other) &&
            this.amplitude === other.getAmplitude() &&
            this.omega === other.getOmega() &&
            this.phase === other.getPhase()
        );
    }

    /** 形如 `2·e^{i(3x + 0.5)}`。 */
    public toString(): string {
        return `${this.amplitude}·e^{i(${this.omega}x + ${this.phase})}`;
    }
}

/** 由 `Complex` 系数构造：`c·e^{iωx}`，其中 `c` 的模为振幅、辐角为相位。 */
export const epicycleFromCoefficient: (coefficient: Complex, omega: number) => Epicycle = (
    coefficient: Complex,
    omega: number
): Epicycle => {
    if (!isComplex(coefficient)) {
        throw new TypeError("Coefficient must be a Complex");
    }
    return new Epicycle(coefficient.magnitude(), omega, coefficient.argument());
};
