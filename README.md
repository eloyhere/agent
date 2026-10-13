# agent

An agent built on the periodicity mathematics of **_Periodicity Mathematics_**.

TypeScript, Vite, Vitest. Strict mode on.

> **Status: foundation migrated, pipeline migration in progress.**
> This section is kept honest on purpose — see "What is not here yet" below.

## What exists now

```
src/validation/     type.ts            runtime type predicates (isNumber, isTarget, …)
src/mathematics/    complex.ts         Complex number class (immutable, radians throughout)
                    epicycle.ts        Epicycle — one complex exponential
                    waveform.ts        Waveform — a sum of epicycles plus a polynomial part
                    operators.ts       the book's operators acting on the two classes above
tests/mathematics/  epicycle.test.ts   21 tests
                    operators.test.ts  18 tests
```

`npm run typecheck` (strict, no errors) and `npm test` (39 passing) are both green.

### `Epicycle` and `Waveform` as complex functions

```
E(x) = A·e^{i(ωx + φ)}
W(x) = Σₙ pₙxⁿ  +  Σₖ Aₖ·e^{i(ωₖx + φₖ)}
```

Both expose the same four entry points:

| call | returns |
|---|---|
| `derivate()` | a **new complex function** of the same kind |
| `derivate(x)` | the derivative **at x**, as a `Complex` |
| `integral()` | a **new complex function** of the same kind |
| `integral(a, b)` | the definite integral, as a `Complex` |

The maths, for a single epicycle:

```
E'(x)  = iω·E(x)     = |ω|A · e^{i(ωx + φ + sgn(ω)·π/2)}
∫E dx  = E(x)/(iω)   = (A/|ω|) · e^{i(ωx + φ − sgn(ω)·π/2)}
```

so differentiating and integrating only act on `(amplitude, phase)` while the
frequency is untouched.

Two design decisions are load-bearing:

1. **`Waveform` carries a polynomial part.** A constant integrates to a linear
   term, a linear term to a quadratic — neither is an epicycle. Without the
   polynomial part, `integral()` could not return "a new complex function" as
   specified, and would have to throw. With it, differentiation and integration
   are **closed in both directions** and neither method has an error path.

2. **`Epicycle` rejects `ω = 0`.** A zero-frequency term *is* a constant, which
   belongs to the polynomial part. Rejecting it keeps `Epicycle.integral()`
   total rather than leaving a corner that throws.

### The book's identities are the tests

`tests/` pins the book's own results, and two of them found real bugs during
migration rather than confirming what was already written:

| identity | source | result |
|---|---|---|
| `Φ_{π/2, 2π}(sin) = −i·cos` | `ex:phase_functor_action` | 1e-10 |
| `Φ_{π, 2π}(sin) = sin` | same example | 1e-10 |
| `W.integral().derivate() = W` | `ch13_adjunction_numeric_example` | 1e-10 |
| `W.derivate().integral() = W − W(0)` | same example | 1e-10 |
| `g(t) = cos(2πt/3), g(0) = 1 ⟹ (I∘D)g = g − 1` | same example, verbatim | 1e-10 |
| `E(translate(W,τ), x) = E(W, x − τ)` | Translation | 1e-10 |
| `T_mn = c_m·conj(d_n)`, diagonal `= Aₖ²` | `def:tendency_convolution` | 1e-12 |
| `Ω = ω₁+ω₂`, `Φ = φ₁+φ₂` | `def:periodic_convolution` | 1e-12 |
| `d = π/2: Sim⁽¹⁾→0, Sim⁽³⁾→−1` | `prop:inflection_periodic_transform` | 1e-14 |
| `d = π: Sim⁽¹⁾→−1, Sim⁽³⁾→+1` | same | 1e-14 |

The `Waveform.integral()` convention was wrong on the first attempt — the
polynomial antiderivative vanishes at 0 while the epicyclic one does not, so
`I∘D` missed the book's `−W(0)`. The test above is what caught it.

## What is not here yet

The reference implementation is the sibling JavaScript project
`R:\project\periodicity-agent` (19 source files, ~4000 lines, 55 passing tests).
Everything below still has to be migrated to TypeScript:

```
mathematics/   fft  linalg  spectrum  periods  intercalation
analysis/      epicycles (fitting)  decision  io  index
learning/      activation (CEAF)  kdi (KDI + baselines)  net  train
token/         bpe  encode (rebuilt on top of Waveform)  index
tests/         the existing 55 tests, ported to Vitest
src/main.ts    replace the Vite template demo with something real
```

Two results from the JavaScript project are worth carrying over, because they
shape the design rather than decorate it:

* **The data-analysis front-end is what makes the learner work.** Initialising a
  CEAF's frequencies from the periodogram instead of at random moves held-out
  R² from ≈0 to **0.9985** and collapses the seed variance to ~0. The book's
  `def:frequency-modes` says exactly this; it is the strongest supported claim in
  the whole exercise.
* **KDI's claimed advantage is not supported.** Over a 54-point
  hyper-parameter sweep it stayed worse than Adam (test MSE 0.0252 vs 0.0010),
  and its best configuration switches off the very term that distinguishes it.

## Commands

```bash
npm install
npm run dev         # Vite dev server
npm run typecheck   # tsc --noEmit, strict
npm test            # vitest run
npm run build       # tsc && vite build
```

## Licence

Apache-2.0, matching the book.
