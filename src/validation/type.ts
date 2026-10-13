/**
 * 运行期类型判定工具。
 *
 * 本模块只做**判定**，不做转换：所有函数都是纯函数，且一律返回类型谓词
 * （`target is T`），以便调用方在 `if` 分支内直接完成类型收窄。
 *
 * 约定：
 *   1. `isNumber` 拒绝 `NaN`，但**允许 ±Infinity** —— 无穷在复分析里是有意义的
 *      取值（例如极点附近的模），而 `NaN` 只可能来自错误的运算，应当尽早拦下。
 *   2. `isTarget` 是**基于 Symbol 标记的实例识别**，配合 `Object.defineProperty`
 *      以不可枚举方式挂上的标记使用。之所以不用 `instanceof`：跨模块副本
 *      （例如打包后被复制成两份）会让 `instanceof` 失效，而 Symbol.for 是
 *      全局注册表，同一 key 在任何副本里都是同一个 Symbol。
 *   3. 函数不接受 `null` / `undefined` 之外的任何隐式转换。
 */

/** 是否为「有意义的」数值：`typeof` 为 number 且非 `NaN`。 */
export const isNumber: (target: unknown) => target is number = (
    target: unknown
): target is number => typeof target === "number" && !Number.isNaN(target);

/** 是否为整数（含 ±Infinity 之外的所有安全整数判定交给调用方）。 */
export const isInteger: (target: unknown) => target is number = (
    target: unknown
): target is number => isNumber(target) && Number.isInteger(target);

/** 是否为正数（严格大于 0，不含 0）。 */
export const isPositive: (target: unknown) => target is number = (
    target: unknown
): target is number => isNumber(target) && target > 0;

/** 是否为有限数。 */
export const isFiniteNumber: (target: unknown) => target is number = (
    target: unknown
): target is number => isNumber(target) && Number.isFinite(target);

/** 是否为字符串。 */
export const isString: (target: unknown) => target is string = (
    target: unknown
): target is string => typeof target === "string";

/** 是否为函数。 */
export const isFunction: (target: unknown) => target is (...args: never[]) => unknown = (
    target: unknown
): target is (...args: never[]) => unknown => typeof target === "function";

/** 是否为非 null 的对象（数组与函数不算，需要它们请用各自的判定）。 */
export const isObject: (target: unknown) => target is Record<PropertyKey, unknown> = (
    target: unknown
): target is Record<PropertyKey, unknown> =>
    typeof target === "object" && target !== null && !Array.isArray(target);

/**
 * 是否带有给定的 Symbol 标记。
 *
 * 用 `Reflect.has` 而非 `in`：前者对 Proxy 与原型链的行为更明确，且不会因
 * 目标为原始值而抛错（`in` 对原始值会抛 TypeError）。
 */
export const isTarget: (target: unknown, key: symbol) => boolean = (
    target: unknown,
    key: symbol
): boolean => {
    if (typeof target !== "object" && typeof target !== "function") {
        return false;
    }
    if (target === null) {
        return false;
    }
    return Reflect.has(target, key);
};

/** 是否为字节序列（`Uint8Array`），BPE 分词层用。 */
export const isBytes: (target: unknown) => target is Uint8Array = (
    target: unknown
): target is Uint8Array => target instanceof Uint8Array;

/**
 * 断言为数值，失败时抛出 `TypeError`。
 *
 * 供内部不变量使用：公开 API 若需要「参数必须是数字」的错误信息，
 * 用本函数比手写 `if` 更一致。
 */
export const assertNumber: (target: unknown, name: string) => asserts target is number = (
    target: unknown,
    name: string
): asserts target is number => {
    if (!isNumber(target)) {
        throw new TypeError(`${name} must be a number`);
    }
};
