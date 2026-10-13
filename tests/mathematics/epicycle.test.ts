import { describe, test, expect } from "vitest";
import { Complex, polarAsRadian, isComplex } from "../../src/mathematics/complex";
import { Epicycle, isEpicycle, epicycleFromCoefficient } from "../../src/mathematics/epicycle";
import { Waveform, isWaveform } from "../../src/mathematics/waveform";

/** 数值求导：中心差分。 */
const numericDerivative = (f: (x: number) => Complex, x: number, h = 1e-6): Complex => {
    const a = f(x + h);
    const b = f(x - h);
    return new Complex((a.getReal() - b.getReal()) / (2 * h), (a.getImaginary() - b.getImaginary()) / (2 * h));
};

/** 数值积分：辛普森法。 */
const numericIntegral = (f: (x: number) => Complex, a: number, b: number, n = 20000): Complex => {
    const h = (b - a) / n;
    let re = 0;
    let im = 0;
    for (let i = 0; i <= n; i += 1) {
        const v = f(a + i * h);
        const w = i === 0 || i === n ? 1 : i % 2 === 1 ? 4 : 2;
        re += w * v.getReal();
        im += w * v.getImaginary();
    }
    const factor = h / 3;
    return new Complex(re * factor, im * factor);
};

const expectClose = (actual: Complex, expected: Complex, tol = 1e-9): void => {
    expect(Math.abs(actual.getReal() - expected.getReal())).toBeLessThan(tol);
    expect(Math.abs(actual.getImaginary() - expected.getImaginary())).toBeLessThan(tol);
};

describe("Epicycle", () => {
    test("value(x) is A·e^{i(ωx + φ)}", () => {
        const e = new Epicycle(2.5, 1.7, 0.4);
        for (const x of [0, 1, -2.3, 7.7]) {
            const v = e.value(x);
            expectClose(v, polarAsRadian(1.7 * x + 0.4, 2.5), 1e-12);
        }
    });

    test("rejects a zero frequency with a clear error", () => {
        expect(() => new Epicycle(1, 0, 0)).toThrow(RangeError);
        expect(() => new Epicycle(1, 0, 0)).toThrow(/non-zero omega/);
    });

    test("normalises a negative amplitude into the phase", () => {
        const e = new Epicycle(-3, 2, 0.5);
        expect(e.getAmplitude()).toBe(3);
        expect(e.getPhase()).toBeCloseTo(0.5 + Math.PI, 12);
        expectClose(e.value(0.9), polarAsRadian(2 * 0.9 + 0.5, -3), 1e-12);
    });

    test("derivate() keeps the frequency and scales the amplitude by |ω|", () => {
        for (const omega of [1.3, -0.7, 4, -2.5]) {
            const e = new Epicycle(1.4, omega, 0.2);
            const d = e.derivate();
            expect(d.getOmega()).toBe(omega);
            expect(d.getAmplitude()).toBeCloseTo(Math.abs(omega) * 1.4, 12);
            expect(d.getPhase()).toBeCloseTo(0.2 + (Math.PI / 2) * Math.sign(omega), 12);
        }
    });

    test("derivate(x) matches numerical differentiation", () => {
        const e = new Epicycle(1.4, -0.9, 0.3);
        for (const x of [0, 0.7, -1.9]) {
            expectClose(e.derivate(x), numericDerivative((t) => e.value(t), x), 1e-7);
        }
    });

    test("integral() keeps the frequency and divides the amplitude by |ω|", () => {
        const e = new Epicycle(2, -3, 0.8);
        const F = e.integral();
        expect(F.getOmega()).toBe(-3);
        expect(F.getAmplitude()).toBeCloseTo(2 / 3, 12);
        expect(F.getPhase()).toBeCloseTo(0.8 - (Math.PI / 2) * Math.sign(-3), 12);
    });

    test("integral(0, x) matches numerical integration", () => {
        const e = new Epicycle(1.1, 2.2, -0.6);
        for (const x of [0.5, 2, -1.3]) {
            const lo = Math.min(0, x);
            const hi = Math.max(0, x);
            const sign = x < 0 ? -1 : 1;
            expectClose(e.integral(0, x), numericIntegral((t) => e.value(t), lo, hi).multiply(sign), 1e-9);
        }
    });

    test("D∘I and I∘D are the identity on a single epicycle", () => {
        const e = new Epicycle(1.7, 1.1, -0.25);
        // D∘I: the antiderivative differs by a constant, which the derivative removes
        expect(e.integral().derivate().equals(e)).toBe(true);
        // I∘D: same, because Epicycle.integral() adds no constant
        expect(e.derivate().integral().equals(e)).toBe(true);
    });

    test("period() and the type guard", () => {
        expect(new Epicycle(1, 2, 0).period()).toBeCloseTo(Math.PI, 12);
        expect(isEpicycle(new Epicycle())).toBe(true);
        expect(isEpicycle(new Complex(1, 0))).toBe(false);
        expect(isEpicycle(null)).toBe(false);
    });

    test("epicycleFromCoefficient reads magnitude and argument", () => {
        const c = polarAsRadian(0.9, 2.5);
        const e = epicycleFromCoefficient(c, 1.4);
        expect(e.getAmplitude()).toBeCloseTo(2.5, 12);
        expect(e.getPhase()).toBeCloseTo(0.9, 12);
        expect(isComplex(c)).toBe(true);
    });
});

describe("Waveform", () => {
    test("value(x) is the sum of the polynomial and the epicycles", () => {
        const w = new Waveform({
            polynomial: [new Complex(1, 0), new Complex(0, 2)],
            epicycles: [new Epicycle(2, 1.5, 0), new Epicycle(0.5, -3, 0.7)]
        });
        for (const x of [0, 1.2, -0.8]) {
            const want = new Complex(1 + 2 * 0, 2 * x)
                .add(new Epicycle(2, 1.5, 0).value(x))
                .add(new Epicycle(0.5, -3, 0.7).value(x));
            expectClose(w.value(x), want, 1e-12);
        }
    });

    test("vanishes identically when it should", () => {
        expect(Waveform.zero().value(3.3).magnitude()).toBe(0);
    });

    test("derivate(x) matches numerical differentiation", () => {
        const w = new Waveform({
            polynomial: [new Complex(0.5, -1), new Complex(2, 0)],
            epicycles: [new Epicycle(1.3, 2.1, 0.2), new Epicycle(0.7, -1.4, -0.9)]
        });
        for (const x of [0, 0.8, -1.6]) {
            expectClose(w.derivate(x), numericDerivative((t) => w.value(t), x), 1e-7);
        }
    });

    test("integral(0, x) matches numerical integration", () => {
        const w = new Waveform({
            polynomial: [new Complex(1, 0.5)],
            epicycles: [new Epicycle(1.2, 1.9, 0.3)]
        });
        for (const x of [0.4, 1.7, -1.1]) {
            const lo = Math.min(0, x);
            const hi = Math.max(0, x);
            const sign = x < 0 ? -1 : 1;
            expectClose(w.integral(0, x), numericIntegral((t) => w.value(t), lo, hi).multiply(sign), 1e-9);
        }
    });

    test("integral() returns the antiderivative normalised to F(0) = 0", () => {
        const w = new Waveform({
            polynomial: [new Complex(2, 1)],
            epicycles: [new Epicycle(1.5, 2.2, 0.4)]
        });
        const F = w.integral();
        expect(F.value(0).magnitude()).toBeLessThan(1e-12);
    });

    test("D∘I = id and I∘D = id − evaluation at 0 (the book's adjunction identities)", () => {
        const w = new Waveform({
            polynomial: [new Complex(0.7, 0.2), new Complex(-1.3, 0.9), new Complex(0.4, -0.1)],
            epicycles: [new Epicycle(1.1, 2.3, 0.35), new Epicycle(0.6, -1.7, -0.8), new Epicycle(2.0, 0.9, 1.2)]
        });
        // D∘I = id exactly, as functions
        const DofI = w.integral().derivate();
        for (const x of [0, 0.6, -1.4, 3.1]) {
            expectClose(DofI.value(x), w.value(x), 1e-10);
        }
        // I∘D = id − W(0)
        const IofD = w.derivate().integral();
        const atZero = w.value(0);
        for (const x of [0, 0.6, -1.4, 3.1]) {
            expectClose(IofD.value(x), w.value(x).subtract(atZero), 1e-10);
        }
    });

    test("the book's own numeric example: g(t)=cos(2πt/3) gives (I∘D)g = g − 1", () => {
        // the book's ch13_adjunction_numeric_example, part (ii):
        //   g(t) = cos(2πt/3), T = 3, g(0) = 1, and (I∘D)g(t) = g(t) − 1
        const omega = (2 * Math.PI) / 3;
        const g = Waveform.fromEpicycles([
            new Epicycle(0.5, omega, 0),
            new Epicycle(0.5, -omega, 0)
        ]);
        // g really is the cosine
        for (const t of [0, 0.45, 1.2, 2.7]) {
            expect(g.value(t).getReal()).toBeCloseTo(Math.cos((2 * Math.PI * t) / 3), 10);
            expect(Math.abs(g.value(t).getImaginary())).toBeLessThan(1e-12);
        }
        expect(g.value(0).getReal()).toBeCloseTo(1, 12);
        // (I∘D)g(t) = g(t) − 1
        const IofD = g.derivate().integral();
        for (const t of [0, 0.45, 1.2, 2.7]) {
            expectClose(IofD.value(t), g.value(t).subtract(g.value(0)), 1e-10);
            expect(IofD.value(t).getReal()).toBeCloseTo(Math.cos((2 * Math.PI * t) / 3) - 1, 10);
        }
        // and the sine case from part (i), where g(0) = 0 so I∘D = g
        const f = Waveform.fromEpicycles([
            new Epicycle(0.5, omega, -Math.PI / 2),
            new Epicycle(0.5, -omega, Math.PI / 2)
        ]);
        expect(Math.abs(f.value(0).getReal())).toBeLessThan(1e-12);
        const IofDf = f.derivate().integral();
        for (const t of [0, 0.45, 1.2]) {
            expectClose(IofDf.value(t), f.value(t), 1e-10);
        }
    });

    test("integration raises the polynomial degree, differentiation lowers it", () => {
        const w = Waveform.constant(new Complex(2, 0));
        expect(w.degree()).toBe(0);
        const F = w.integral();
        expect(F.degree()).toBe(1);
        expect(F.derivate().degree()).toBe(0);
        // double integration is still representable, so the class stays closed
        expect(F.integral().degree()).toBe(2);
    });

    test("trailing zero coefficients are trimmed, so equals is structural", () => {
        const a = new Waveform({ polynomial: [new Complex(1, 0), new Complex(0, 0)] });
        const b = new Waveform({ polynomial: [new Complex(1, 0)] });
        expect(a.equals(b)).toBe(true);
        expect(a.degree()).toBe(0);
        expect(Waveform.zero().degree()).toBe(-1);
    });

    test("plus and times behave as expected", () => {
        const a = new Waveform({ epicycles: [new Epicycle(1, 2, 0)] });
        const b = new Waveform({ polynomial: [new Complex(1, 1)] });
        const sum = a.plus(b);
        for (const x of [0, 1.5]) {
            expectClose(sum.value(x), a.value(x).add(b.value(x)), 1e-12);
        }
        const scaled = a.times(new Complex(0, 3));   // 3i
        for (const x of [0, 1.5]) {
            expectClose(scaled.value(x), a.value(x).multiply(new Complex(0, 3)), 1e-12);
        }
    });

    test("type guards and validation", () => {
        expect(isWaveform(new Waveform())).toBe(true);
        expect(isWaveform(new Epicycle())).toBe(false);
        expect(() => new Waveform({ epicycles: [new Complex(1, 0)] as unknown as Epicycle[] })).toThrow(TypeError);
        expect(() => new Waveform({ polynomial: [1 as unknown as Complex] })).toThrow(TypeError);
    });
});
