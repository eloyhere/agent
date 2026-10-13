# agent

A conversational agent built on the periodicity mathematics of
**_Periodicity Mathematics_**.

TypeScript, Vite, Vitest. Strict mode. **Zero runtime dependencies** — the FFT,
the eigensolvers, the BPE trainer and the dialogue engine are all in this repo.

```bash
npm install
npm run dev         # chat UI at the printed URL
npm test            # 59 tests
npm run typecheck   # tsc --noEmit, strict
npm run build
```

## What it can talk about

| you say | it does |
|---|---|
| a numeric series (≥ 8 values) | periodicity diagnosis: autocorrelation, periodogram, R/S Hurst, cyclic autocorrelation, and the seven-type classification of `cor:data_period_type` |
| some text | byte-level BPE tokenisation, then each token is shown as an **epicycle group** |
| "how long should the window be", or a year length and a month length | continued fractions, convergents, the leakage bound, the Metonic cycle |
| "what is leakage" / "explain hurst" / … | concept answers, with the corrections found during verification |

**Honest boundary.** The language layer is a local template engine. It answers
fluently about periodicity, tokenisation, intercalation and the book's theorems,
and it remembers the conversation — but it does **not** do small talk or
unrelated questions, and it says so rather than pretending. See
[Plugging in a real language model](#plugging-in-a-real-language-model).

## The idea: a conversation is a Waveform

The book already has the object. `ex:dft_epicycle` says the DFT corresponds to the
finite epicycle group

```
E_f = { ( |f_hat(k)| / T0 ,  2 pi k / T0 ,  arg f_hat(k) ) }      k = 0 .. T0-1
```

and `def:eaf_agent`'s CEAF is its sum. So a token is encoded as one: take its
bytes, centre them, and treat that as one period of a periodic signal of length
`L` — **a word's period is its length**.

```
s(n) = b_n - 128,   n = 0 .. L-1
c_k  = (1/L) sum_n s(n) e^{-i 2 pi k n / L}
token = { period: L,  dc: A_0 e^{i phi_0},  atoms: (A_k, 2 pi k / L, phi_k) }
```

If every utterance is a `Waveform`, then the whole conversation is one too:

| conversational notion | the periodicity object |
|---|---|
| position in the dialogue | **Translation** — phase rotates by `-omega_k * tau`, a *frequency-dependent* rotation, so position is native here rather than bolted on |
| mixing context | **Periodic convolution** — `Omega = omega1 + omega2`, `Phi = phi1 + phi2` |
| "what have we been talking about" | the **trend-convolution diagonal** `A_k^2`, plotted live in the side panel |
| word length | **Period** |
| word weight | **Scaling** |

## Layout

```
src/mathematics/  complex.ts     Complex (immutable, radians throughout)
                  epicycle.ts    Epicycle — one complex exponential
                  waveform.ts    Waveform — sum of epicycles plus a polynomial part
                  operators.ts   the book's operators acting on both
                  fft.ts         radix-2 plus Bluestein, for arbitrary lengths
src/analysis/     periods.ts     autocorrelation (1/T), R/S Hurst, cyclic autocorrelation
                  spectrum.ts    periodogram, peak refinement, windowed kernel
                  intercalation.ts  continued fractions, convergents, leakage bound
                  analyse.ts     the chapter-31 pipeline and the seven-type decision
src/token/        bpe.ts         byte-level BPE (no out-of-vocabulary input)
                  encode.ts      token -> Waveform
src/dialogue/     language-model.ts  the LanguageModel seam
                  agent.ts       intents, tools, bilingual rendering, dialogue state
src/main.ts       the chat UI
tests/            59 tests
```

## `Epicycle` and `Waveform` as complex functions

```
E(x) = A e^{i(omega x + phi)}
W(x) = sum_n p_n x^n  +  sum_k A_k e^{i(omega_k x + phi_k)}
```

Both expose the same four entry points:

| call | returns |
|---|---|
| `derivate()` | a **new complex function** of the same kind |
| `derivate(x)` | the derivative **at x**, as a `Complex` |
| `integral()` | a **new complex function** of the same kind |
| `integral(a, b)` | the definite integral, as a `Complex` |

For a single epicycle:

```
E'(x)  = i omega E(x)   = |omega| A e^{i(omega x + phi + sgn(omega) pi/2)}
int E  = E(x)/(i omega) = (A/|omega|) e^{i(omega x + phi - sgn(omega) pi/2)}
```

so differentiating and integrating only move `(amplitude, phase)`; the frequency is
untouched.

Two decisions are load-bearing:

1. **`Waveform` carries a polynomial part.** A constant integrates to a linear
   term, a linear term to a quadratic — neither is an epicycle. Without it
   `integral()` could not return a new function of the same kind and would have to
   throw. With it, differentiation and integration are **closed in both
   directions** and neither method has an error path.
2. **`Epicycle` rejects `omega = 0`.** A zero-frequency term *is* a constant, which
   belongs to the polynomial part. Rejecting it keeps `Epicycle.integral()` total.

## Verified against the book's own results

The tests pin the book's identities rather than restating the implementation. Two
of them caught real bugs during development.

| identity | source | result |
|---|---|---|
| `Phi_{pi/2, 2pi}(sin) = -i cos` | `ex:phase_functor_action` | 1e-10 |
| `Phi_{pi, 2pi}(sin) = sin` | same | 1e-10 |
| `W.integral().derivate() = W` | `ch13_adjunction_numeric_example` | 1e-10 |
| `W.derivate().integral() = W - W(0)` | same | 1e-10 |
| `g = cos(2 pi t/3), g(0)=1 => (I o D)g = g - 1` | same, verbatim | 1e-10 |
| `E(translate(W,tau), x) = W(x - tau)` | Translation | 1e-10 |
| `T_mn = c_m conj(d_n)`, diagonal `= A_k^2` | `def:tendency_convolution` | 1e-12 |
| `Omega = omega1+omega2`, `Phi = phi1+phi2` | `def:periodic_convolution` | 1e-12 |
| `d = pi/2: Sim(1)->0, Sim(3)->-1` | `prop:inflection_periodic_transform` | 1e-14 |
| byte-exact BPE round trip, incl. Chinese and emoji | — | exact |
| `[1;2,2,...]` convergents `1/1, 3/2, 7/5, 17/12, 41/29` | — | exact |
| Metonic: 235 months in 19 years; 7th convergent **4131/334** | `ex:metonic_parallel` | exact |

### Three formulas the book states incorrectly, implemented in their corrected form

Found by the numerical-verification pass against both editions, and fixed in the
book. The regressions are pinned in the tests so the code cannot drift back.

1. **The windowed spectrum is a convolution, not a product** (`defect 7`). The
   book wrote `w_hat(...) * D_T(...)`; the correct single kernel is
   `W_T(theta) = (1/T) sum_n w[n] e^{i 2 pi n theta}`. The product form is off by
   1.210 relative for a Hann window.
2. **The autocorrelation needs the `1/T` normalisation** (`defect 9`). The
   unnormalised sum is `T` times larger, and that factor had been copied into 14
   chapters.
3. **The Metonic seventh convergent is `4131/334`, not `3896/315`** (`defect 13`).
   Verified with mpmath at 60 digits against both the book's own decimals and
   modern IAU values.

### Two bugs the tests caught here, not by inspection

* `findPeaks` excluded the Nyquist bin, so a **period-2 alternating series** — the
  purest periodic signal there is — was invisible to the analyser.
* `periodicConvolve` omitted the **DC cross-terms**. Convolving with a purely
  constant waveform (a space token, say) produced an empty cross product and wiped
  out every epicycle, when multiplying by a constant must not change the frequency
  content. The test that accumulates a real conversation is what exposed it.

## Plugging in a real language model

The dialogue layer depends only on this interface:

```ts
interface LanguageModel {
  readonly name: string;
  readonly isLocal: boolean;
  complete(request: CompletionRequest): Promise<string>;
}
```

`TemplateLanguageModel` is the local, offline default. `HttpLanguageModel` is a
vendor-neutral skeleton: endpoint, model name and key are all supplied by the
caller, nothing is hardcoded. Swapping the layer changes **nothing** in the intent
parser, the tools or the dialogue state.

Note that a browser calling a third-party API directly is usually blocked by
**CORS**; in that case point the endpoint at a same-origin forwarder.

## Two results from the wider project that shape the design

* **The data-analysis front-end is what makes the learner work.** Initialising a
  CEAF's frequencies from the periodogram rather than at random moves held-out R²
  from ≈0 to **0.9985** and collapses the seed variance to ~0. The book's
  `def:frequency-modes` says exactly this; it is the strongest supported claim in
  the whole exercise.
* **KDI's claimed advantage is not supported.** Over a 54-point hyper-parameter
  sweep it stayed worse than Adam (test MSE 0.0252 vs 0.0010), and its best
  configuration switches off the very term that distinguishes it. That is a
  negative result on the tasks tried, not a proof that KDI is useless — the
  JavaScript reference records the full numbers.

## Licence

Apache-2.0, matching the book.
