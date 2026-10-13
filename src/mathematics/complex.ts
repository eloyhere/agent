import { isNumber, isTarget } from "../validation/type";

const Identity: Record<string, symbol> = Object.freeze({
    Complex: Symbol.for("Complex")
});

/** 是否为本模块的复数实例。 */
export const isComplex: (target: unknown) => target is Complex = (
    target: unknown
): target is Complex => isTarget(target, Identity.Complex);

interface PolarAsDegree {
    (target: number): Complex;
    (target: number, magnitude: number): Complex;
}

/**
 * 由极坐标构造复数：`target` 为辐角（度），`magnitude` 为模（默认 1）。
 *
 * 度与弧度的换算只在这里发生，其余方法一律以弧度为准。
 */
export const polarAsDegree: PolarAsDegree = (target: number, magnitude: number = 1): Complex => {
    if (!isNumber(target) || !isNumber(magnitude)) {
        throw new TypeError("Argument and magnitude must be numbers");
    }
    return new Complex(
        magnitude * Math.cos((target * Math.PI) / 180),
        magnitude * Math.sin((target * Math.PI) / 180)
    );
};

interface PolarAsRadian {
    (target: number): Complex;
    (target: number, magnitude: number): Complex;
}

/** 由极坐标构造复数：`target` 为辐角（弧度），`magnitude` 为模（默认 1）。 */
export const polarAsRadian: PolarAsRadian = (target: number, magnitude: number = 1): Complex => {
    if (!isNumber(target) || !isNumber(magnitude)) {
        throw new TypeError("Argument and magnitude must be numbers");
    }
    return new Complex(magnitude * Math.cos(target), magnitude * Math.sin(target));
};

/**
 * 复数。
 *
 * 设计要点：
 *   1. **全部运算返回新实例**，不修改自身，故可安全共享与缓存。
 *   2. **辐角统一以弧度表示**（`argument()`），与 `Math.atan2` 一致；
 *      需要度数时由调用方自行换算，或改用 `polarAsDegree` 构造。
 *   3. **与实数的四则运算**按「实部参与运算、虚部原样保留」处理，
 *      即把实数视作实轴上的平移/缩放，而非把它当作 (c, 0) 参与复数乘法 ——
 *      后者会让 `multiply(2)` 得到虚部翻倍的意外结果。
 *   4. 除法用共轭形式展开，分母取模的平方，避免虚数单位除法。
 */
export class Complex {
    protected real: number;
    protected imaginary: number;
    /**
     * 类型标记，供 isTarget 在运行期识别本类实例。
     *
     * 必须声明为可被 Reflect 探测的普通属性，不能用 `#` 私有字段：
     * 私有字段是内部槽而非属性，Reflect.has / Reflect.ownKeys 一概看不见，
     * 用它会让 isComplex 恒为 false，进而所有 Complex 入参都被判为非法。
     * 用 declare 只为占位类型，实际属性在构造函数里以不可枚举方式定义（见下）。
     */
    declare readonly [Identity.Complex]: symbol;

    public constructor(real: number = 0, imaginary: number = 0) {
        if (!isNumber(real) || !isNumber(imaginary)) {
            throw new TypeError("Real and imaginary parts must be numbers");
        }
        this.real = real;
        this.imaginary = imaginary;
        // 以不可枚举方式挂上类型标记：键必须是 Identity.Complex 这个 Symbol 本身，
        // 因为 isComplex 走的是 Reflect.has(target, Identity.Complex)，字符串键探不到。
        // 不可枚举则保证它不出现在 Object.keys / JSON.stringify 里，不污染对外数据结构。
        Object.defineProperty(this, Identity.Complex, {
            value: Identity.Complex,
            enumerable: false,
            writable: false,
            configurable: false
        });
    }

    public getReal(): number {
        return this.real;
    }

    public getImaginary(): number {
        return this.imaginary;
    }

    /** 模 |z| = √(a² + b²)。 */
    public magnitude(): number {
        return Math.sqrt(this.real * this.real + this.imaginary * this.imaginary);
    }

    /** 辐角 arg(z) = atan2(b, a)，弧度，范围 (-π, π]。 */
    public argument(): number {
        return Math.atan2(this.imaginary, this.real);
    }

    /** 共轭 a - bi。 */
    public conjugate(): Complex {
        return new Complex(this.real, -this.imaginary);
    }

    /** 相反数 -a - bi。 */
    public negate(): Complex {
        return new Complex(-this.real, -this.imaginary);
    }

    /** 与另一点的欧氏距离 |z - w|。 */
    public distance(other: Complex): number {
        return Math.hypot(this.real - other.getReal(), this.imaginary - other.getImaginary());
    }

    /** 两点辐角之差 arg(other) - arg(this)，弧度。 */
    public angular(other: Complex): number {
        return other.argument() - this.argument();
    }

    public add(other: number): Complex;
    public add(other: Complex): Complex;
    public add(other: number | Complex): Complex {
        if (isNumber(other)) {
            return new Complex(this.real + other, this.imaginary + other);
        }
        if (isComplex(other)) {
            return new Complex(this.real + other.getReal(), this.imaginary + other.getImaginary());
        }
        throw new TypeError("Value must be a number or a Complex");
    }

    public subtract(other: number): Complex;
    public subtract(other: Complex): Complex;
    public subtract(other: number | Complex): Complex {
        if (isNumber(other)) {
            return new Complex(this.real - other, this.imaginary - other);
        }
        if (isComplex(other)) {
            return new Complex(this.real - other.getReal(), this.imaginary - other.getImaginary());
        }
        throw new TypeError("Value must be a number or a Complex");
    }

    public multiply(other: number): Complex;
    public multiply(other: Complex): Complex;
    public multiply(other: number | Complex): Complex {
        if (isNumber(other)) {
            return new Complex(this.real * other, this.imaginary * other);
        }
        if (isComplex(other)) {
            const a = this.real;
            const b = this.imaginary;
            const c = other.getReal();
            const d = other.getImaginary();
            return new Complex(a * c - b * d, a * d + b * c);
        }
        throw new TypeError("Value must be a number or a Complex");
    }

    public divide(other: number): Complex;
    public divide(other: Complex): Complex;
    public divide(other: number | Complex): Complex {
        if (isNumber(other)) {
            if (other === 0) {
                throw new RangeError("Division by zero");
            }
            return new Complex(this.real / other, this.imaginary / other);
        }
        if (isComplex(other)) {
            const c = other.getReal();
            const d = other.getImaginary();
            const denominator = c * c + d * d;
            if (denominator === 0) {
                throw new RangeError("Division by zero");
            }
            const a = this.real;
            const b = this.imaginary;
            return new Complex((a * c + b * d) / denominator, (b * c - a * d) / denominator);
        }
        throw new TypeError("Value must be a number or a Complex");
    }

    /**
     * 主值对数：Log z = ln|z| + i · arg(z)。
     *
     * 与实对数一样，必须是**单值**函数才谈得上「主值」：
     * 幅角取主值 (-π, π]，故 z 的辐角每绕一圈，对数的虚部就差 2π。
     * 因此 Log z 只在割线（负实轴，含原点）外解析，这正是复对数一般不能逐点相乘的原因：
     *   Log(z·w) ≠ Log z + Log w，两者可能相差 2πi。
     *
     * 单位：虚部是**弧度**（与 argument() / Math.atan2 一致）。
     * 本模块是复分析层，全程弧度；要度数请自行换算，勿把项目的度数约定带进来。
     * z = 0 时对数无定义，抛 RangeError。
     */
    public log(): Complex {
        if (this.real === 0 && this.imaginary === 0) {
            throw new RangeError("Cannot take the logarithm of zero");
        }
        return new Complex(Math.log(this.magnitude()), this.argument());
    }

    /**
     * 对数：本复数的**主值对数**乘以给定因子。
     *
     * 记法源自 z^w = e^(w · Log z)：本方法给出的正是指数里那个 w · Log z。
     * 故 logarithm(other) 的主部对应该分支下 z^other 的辐角。
     * 传入复数即按复数乘法（w · Log z）计算。
     */
    public logarithm(factor: number): Complex;
    public logarithm(factor: Complex): Complex;
    public logarithm(factor: number | Complex): Complex {
        if (isNumber(factor)) {
            return this.log().multiply(factor);
        }
        if (isComplex(factor)) {
            return this.log().multiply(factor);
        }
        throw new TypeError("Value must be a number or a Complex");
    }

    /**
     * 幂：z^w。
     *
     * 实数指数走两条路，以求精确：
     *   * **整数**指数用平方快速幂（负整数先取倒数），结果为精确的代数运算，
     *     且 z 为 0 的负数次幂会抛错 —— 与 lower 的整数语义保持一致；
     *   * **非整数**实数指数按多值函数的**主值**分支计算：z^w = e^(w · Log z)，
     *     故 z 为 0 且 w ≤ 0 时抛错。
     *
     * 复数指数一律走主值分支 z^w = e^(w · Log z)。
     * 因主值对数有割线，故 (−1)^0.5 得 i（主值），而非一般多值意义下的 ±i ——
     * 需要全部 n 个 n 次方根时，应由调用方另行按辐角加 2kπ/n 展开。
     *
     * 约定 0^0 = 1（与「空乘积为 1」的代数惯例一致），其余 0 的非正数次幂抛错。
     */
    public power(exponent: number): Complex;
    public power(exponent: Complex): Complex;
    public power(exponent: number | Complex): Complex {
        if (isNumber(exponent)) {
            if (!Number.isInteger(exponent)) {
                return this.principalPower(exponent);
            }
            if (exponent === 0) {
                return new Complex(1, 0);
            }
            if (exponent < 0) {
                return new Complex(1, 0).divide(this.power(-exponent));
            }
            // 显式标注为 Complex：若由初始值推断，`result` 会带上多态 this 类型，
            // 后续赋值 Complex 实例即报 TS2322。幂运算本就只保证返回 Complex，不需要保留子类信息。
            let result: Complex = new Complex(1, 0);
            let base: Complex = new Complex(this.real, this.imaginary);
            let remaining: number = exponent;
            while (remaining > 0) {
                if (remaining % 2 === 1) {
                    result = result.multiply(base);
                }
                base = base.multiply(base);
                remaining = Math.floor(remaining / 2);
            }
            return result;
        }
        if (isComplex(exponent)) {
            return this.principalPower(exponent);
        }
        throw new TypeError("Exponent must be a number or a Complex");
    }

    /**
     * 主值分支上的幂：z^w = e^(w · Log z)。
     *
     * 0^w 只在 w = 0 时有意义（约定为 1，见 power 注释），
     * 其余情形 Log 0 无定义，故抛 RangeError。
     */
    private principalPower(exponent: number | Complex): Complex {
        const zero: boolean = this.real === 0 && this.imaginary === 0;
        if (zero) {
            if (isNumber(exponent) && exponent === 0) {
                return new Complex(1, 0);
            }
            if (isComplex(exponent) && exponent.getReal() === 0 && exponent.getImaginary() === 0) {
                return new Complex(1, 0);
            }
            throw new RangeError(
                "Cannot raise zero to a non-zero power via the principal logarithm"
            );
        }
        // 分支收窄后再调 multiply：multiply 是重载方法，传 number | Complex 联合类型匹配不上任何一支。
        const product: Complex = isNumber(exponent)
            ? this.log().multiply(exponent)
            : this.log().multiply(exponent);
        return polarAsRadian(product.getImaginary(), Math.exp(product.getReal()));
    }

    /** 按分量判等（严格相等，不做容差）。 */
    public equals(other: Complex): boolean {
        return (
            isComplex(other) &&
            this.real === other.getReal() &&
            this.imaginary === other.getImaginary()
        );
    }

    /** 形如 `3 + 4i` / `3 - 4i` / `-2i`。 */
    public toString(): string {
        if (this.imaginary === 0) {
            return `${this.real}`;
        }
        if (this.real === 0) {
            return `${this.imaginary}i`;
        }
        return this.imaginary > 0
            ? `${this.real} + ${this.imaginary}i`
            : `${this.real} - ${-this.imaginary}i`;
    }
}
