import { describe, test, expect } from "vitest";
import { Complex, polarAsRadian } from "../../src/mathematics/complex";
import { Epicycle } from "../../src/mathematics/epicycle";
import { Waveform } from "../../src/mathematics/waveform";
import {
    translate,
    translateEpicycle,
    phaseRotate,
    phaseFunctor,
    scale,
    similarity,
    similarityKernels,
    periodicConvolve,
    trendConvolution,
    trendDiagonal
} from "../../src/mathematics/operators";

const expectClose = (actual: Complex, expected: Complex, tol = 1e-9): void => {
    expect(Math.abs(actual.getReal() - expected.getReal())).toBeLessThan(tol);
    expect(Math.abs(actual.getImaginary() - expected.getImaginary())).toBeLessThan(tol);
};

const sample = (): Waveform =>
    new Waveform({
        polynomial: [new Complex(0.4, -0.2), new Complex(1.1, 0.3)],
        epicycles: [new Epicycle(1.3, 1.7, 0.25), new Epicycle(0.6, -2.4, -0.9)]
    });

describe("Translation", () => {
    test("shifts the argument: W(translate(W,τ), x) = W(x − τ)", () => {
        const w = sample();
        for (const tau of [0, 0.8, -1.7, 3.3]) {
            const shifted = translate(w, tau);
            for (const x of [0, 0.5, 2.1, -1.2]) {
                expectClose(shifted.value(x), w.value(x - tau), 1e-10);
            }
        }
    });

    test("is a frequency-dependent phase rotation, not a global one", () => {
        const e = new Epicycle(2, 3, 0.1);
        const t = translateEpicycle(e, 0.5);
        expect(t.getAmplitude()).toBe(2);
        expect(t.getPhase()).toBeCloseTo(0.1 - 3 * 0.5, 12);
        // a second epicycle at a different frequency must rotate by a different amount
        const e2 = new Epicycle(2, -3, 0.1);
        expect(translateEpicycle(e2, 0.5).getPhase()).toBeCloseTo(0.1 + 3 * 0.5, 12);
    });

    test("leaves the amplitudes alone", () => {
        const w = sample();
        const t = translate(w, 2.2);
        expect(t.getEpicycles().map((e) => e.getAmplitude())).toEqual(
            w.getEpicycles().map((e) => e.getAmplitude())
        );
    });
});

describe("Rotation and scaling", () => {
    test("phaseRotate multiplies by e^{iθ}", () => {
        const w = sample();
        const theta = 0.63;
        const r = phaseRotate(w, theta);
        for (const x of [0, 1.4, -0.7]) {
            expectClose(r.value(x), w.value(x).multiply(polarAsRadian(theta)), 1e-10);
        }
    });

    test("scale multiplies by a real factor", () => {
        const w = sample();
        const s = scale(w, -2.5);
        for (const x of [0, 1.1]) {
            expectClose(s.value(x), w.value(x).multiply(-2.5), 1e-10);
        }
    });
});

describe("the book's phase functor", () => {
    test("Phi_{θ,T}(f)(t) = e^{iθ} f(t − θT/2π)", () => {
        const w = sample();
        const theta = 0.9;
        const period = 3.2;
        const f = phaseFunctor(w, theta, period);
        const tau = (theta * period) / (2 * Math.PI);
        for (const t of [0, 0.7, 2.4]) {
            expectClose(f.value(t), w.value(t - tau).multiply(polarAsRadian(theta)), 1e-10);
        }
    });

    test("the book's example: Phi_{π/2, 2π}(sin) = −i·cos", () => {
        // f(t) = sin t is the conjugate pair at ±1 with phase ∓π/2
        const f = Waveform.fromEpicycles([
            new Epicycle(0.5, 1, -Math.PI / 2),
            new Epicycle(0.5, -1, Math.PI / 2)
        ]);
        const period = 2 * Math.PI;
        const theta = Math.PI / 2;
        const got = phaseFunctor(f, theta, period);
        for (const t of [0, 1, 2.5, 4]) {
            // sin t itself, for reference
            expect(f.value(t).getReal()).toBeCloseTo(Math.sin(t), 10);
            // the functor must give −i·cos t
            expectClose(got.value(t), new Complex(0, -Math.cos(t)), 1e-10);
        }
    });

    test("the identity branch: Phi_{π, 2π}(sin) = sin", () => {
        const f = Waveform.fromEpicycles([
            new Epicycle(0.5, 1, -Math.PI / 2),
            new Epicycle(0.5, -1, Math.PI / 2)
        ]);
        const got = phaseFunctor(f, Math.PI, 2 * Math.PI);
        for (const t of [0, 1, 2.5, 4]) {
            expectClose(got.value(t), f.value(t), 1e-10);
        }
    });
});

describe("similarity kernels", () => {
    test("the six kernels follow the book's identities", () => {
        for (const d of [0, 0.3, Math.PI / 2, 1.9, -2.4, Math.PI]) {
            const k1 = similarityKernels(d, 1);
            const k3 = similarityKernels(d, 3);
            const k4 = similarityKernels(d, 4);
            const k5 = similarityKernels(d, 5);
            expect(k1.cosine).toBeCloseTo(Math.cos(d), 14);
            expect(k1.sine).toBeCloseTo(Math.sin(d), 14);
            expect(k3.cosine).toBeCloseTo(Math.cos(2 * d), 14);
            expect(k4.sine).toBeCloseTo(Math.sin(2 * d), 14);
            expect(k5.cosine).toBeCloseTo(Math.cos(d / 2), 14);
            // Pythagorean, double-angle and half-angle consistency
            expect(k1.cosine ** 2 + k1.sine ** 2).toBeCloseTo(1, 12);
            expect(k3.cosine).toBeCloseTo(k1.cosine ** 2 - k1.sine ** 2, 12);
            expect(k4.sine).toBeCloseTo(2 * k1.cosine * k1.sine, 12);
        }
    });

    test("the inflection table: d = π/2 and d = π", () => {
        const a = new Epicycle(1, 1, 0);
        const at = (d: number): Epicycle => new Epicycle(1, 1, d);
        // orthogonal trends
        expect(similarity(a, at(Math.PI / 2), 1)).toBeCloseTo(0, 14);
        expect(similarity(a, at(Math.PI / 2), 3)).toBeCloseTo(-1, 14);
        // direction reversal
        expect(similarity(a, at(Math.PI), 1)).toBeCloseTo(-1, 14);
        expect(similarity(a, at(Math.PI), 2)).toBeCloseTo(0, 14);
        expect(similarity(a, at(Math.PI), 3)).toBeCloseTo(1, 14);
    });

    test("rejects out-of-range kernel indices", () => {
        const a = new Epicycle(1, 1, 0);
        expect(() => similarity(a, a, 0)).toThrow(RangeError);
        expect(() => similarity(a, a, 7)).toThrow(RangeError);
    });
});

describe("periodic convolution", () => {
    test("composes frequencies and phases additively", () => {
        const a = Waveform.fromEpicycles([new Epicycle(1.2, 0.7, 0.31)]);
        const b = Waveform.fromEpicycles([new Epicycle(0.8, 1.9, -0.44)]);
        const out = periodicConvolve(a, b, { useSimilarity: false });
        expect(out.getEpicycles()).toHaveLength(1);
        const e = out.getEpicycles()[0];
        expect(e.getOmega()).toBeCloseTo(0.7 + 1.9, 12);
        expect(e.getPhase()).toBeCloseTo(0.31 - 0.44, 12);
        expect(e.getAmplitude()).toBeCloseTo(1.2 * 0.8, 12);
    });

    test("with the similarity kernel the amplitude is scaled by Sim in [-1, 1]", () => {
        const a = Waveform.fromEpicycles([new Epicycle(1, 2, 0)]);
        const b = Waveform.fromEpicycles([new Epicycle(1, 3, 0.4)]);
        const plain = periodicConvolve(a, b, { useSimilarity: false });
        const withKernel = periodicConvolve(a, b, { useSimilarity: true, d: 1 });
        expect(withKernel.getEpicycles()[0].getAmplitude()).toBeLessThanOrEqual(
            plain.getEpicycles()[0].getAmplitude() + 1e-12
        );
        // Sim^(1) at the phase difference 0 - 0.4
        expect(withKernel.getEpicycles()[0].getAmplitude()).toBeCloseTo(Math.cos(-0.4), 12);
    });

    test("keeps every frequency pair, capped by keep", () => {
        const a = Waveform.fromEpicycles([new Epicycle(1, 1, 0), new Epicycle(1, 2, 0), new Epicycle(1, 3, 0)]);
        const b = Waveform.fromEpicycles([new Epicycle(1, 4, 0), new Epicycle(1, 5, 0)]);
        const out = periodicConvolve(a, b, { useSimilarity: false, keep: 64 });
        expect(out.getEpicycles()).toHaveLength(6);
        const capped = periodicConvolve(a, b, { useSimilarity: false, keep: 4 });
        expect(capped.getEpicycles()).toHaveLength(4);
    });

    test("composes the constant (DC) part multiplicatively", () => {
        const a = Waveform.constant(new Complex(2, 0));
        const b = Waveform.constant(new Complex(0, 3));
        const out = periodicConvolve(a, b, { useSimilarity: false });
        expectClose(out.value(0), new Complex(0, 6), 1e-12);
    });
});

describe("trend convolution", () => {
    test("is T_mn = c_m conj(d_n)", () => {
        const w = sample();
        const t = trendConvolution(w, w);
        const epicycles = w.getEpicycles();
        for (let i = 0; i < t.rows; i += 1) {
            for (let j = 0; j < t.columns; j += 1) {
                const p = epicycles[i];
                const q = epicycles[j];
                const amplitude = p.getAmplitude() * q.getAmplitude();
                const delta = p.getPhase() - q.getPhase();
                expect(t.real[i * t.columns + j]).toBeCloseTo(amplitude * Math.cos(delta), 12);
                expect(t.imaginary[i * t.columns + j]).toBeCloseTo(amplitude * Math.sin(delta), 12);
            }
        }
    });

    test("is Hermitian: T_ij = conj(T_ji)", () => {
        const w = sample();
        const t = trendConvolution(w, w);
        for (let i = 0; i < t.rows; i += 1) {
            for (let j = 0; j < t.columns; j += 1) {
                expect(t.real[i * t.columns + j]).toBeCloseTo(t.real[j * t.columns + i], 12);
                expect(t.imaginary[i * t.columns + j]).toBeCloseTo(-t.imaginary[j * t.columns + i], 12);
            }
        }
    });

    test("the diagonal is A_k squared, real and non-negative", () => {
        const w = sample();
        const diagonal = trendDiagonal(w);
        for (let k = 0; k < diagonal.length; k += 1) {
            expect(diagonal[k]).toBeCloseTo(w.getEpicycles()[k].getAmplitude() ** 2, 12);
            expect(diagonal[k]).toBeGreaterThanOrEqual(0);
        }
    });
});
