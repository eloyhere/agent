import { describe, test, expect } from "vitest";
import { BpeTokenizer } from "../../src/token/bpe";
import { PeriodicityAgent, detectLanguage, extractNumbers, parseIntent } from "../../src/dialogue/agent";
import { analyseSeries, PERIOD_TYPES } from "../../src/analysis/analyse";
import { metonicAnalysis, continuedFraction, convergents, rankWindowLengths } from "../../src/analysis/intercalation";
import { encodeBytes, tokenizeToWaveforms, reconstructBytes } from "../../src/token/encode";

const CORPUS = [
    "periodicity periodic signal sampling window leakage epicycle waveform",
    "intercalation calendar continued fraction convergent denominator hurst",
    "周期性 周期 信号 采样 窗口 泄漏 周转圆 振幅 频率 相位 谐波",
    "置闰 历法 连分数 收敛子 分母 偏移 自相关 周期图 谱 峰 分词"
].join("\n");

const tokenizer = (): BpeTokenizer => {
    const t = new BpeTokenizer();
    t.train(CORPUS, { vocabularySize: 300, maxCorpusBytes: 4000 });
    return t;
};

describe("byte-level BPE", () => {
    test("round-trips ASCII, Chinese and emoji byte-exactly", () => {
        const t = tokenizer();
        for (const s of ["periodic signal", "周转圆组与周期性", "emoji 🙂🚀", "éàü\u0000\u00ff"]) {
            expect(t.decode(t.encode(s))).toBe(s);
        }
    });

    test("compresses relative to raw bytes and is deterministic", () => {
        const a = tokenizer();
        const b = tokenizer();
        expect(a.merges).toEqual(b.merges);
        const s = "periodicity of a periodic signal";
        expect(a.encode(s).length).toBeLessThan(new TextEncoder().encode(s).length);
    });
});

describe("token as an epicycle group", () => {
    test("evaluating the group at integer samples reconstructs the bytes", () => {
        for (const token of ["a", "period", "group", "周转"]) {
            const bytes = new TextEncoder().encode(token);
            const waveform = encodeBytes(bytes);
            const back = reconstructBytes(waveform, bytes.length);
            expect(Array.from(back)).toEqual(Array.from(bytes));
        }
    });

    test("the period of a token is its byte length", () => {
        const t = tokenizer();
        const tokens = tokenizeToWaveforms("periodic 周期", t);
        for (const token of tokens) {
            expect(token.period).toBe(token.bytes.length);
        }
    });

    test("a single byte has no harmonics, only DC", () => {
        const waveform = encodeBytes(Uint8Array.of(65));
        expect(waveform.getEpicycles()).toHaveLength(0);
        expect(waveform.getPolynomial()[0].getReal()).toBeCloseTo(65 - 128, 12);
    });
});

describe("analysis core", () => {
    test("a pure alternating sequence is classified as strictly periodic", () => {
        const values = Array.from({ length: 128 }, (_, i) => (i % 2 === 0 ? 1 : -1));
        const report = analyseSeries(values);
        expect(report.peaks.length).toBeGreaterThan(0);
        // period 2 samples
        expect(report.peaks[0].period).toBeCloseTo(2, 1);
        expect([PERIOD_TYPES.STRICT, PERIOD_TYPES.UNIFORM]).toContain(report.diagnosis.type);
    });

    test("a constant series is trivially periodic", () => {
        const report = analyseSeries(Array.from({ length: 64 }, () => 3));
        expect(report.diagnosis.type).toBe(PERIOD_TYPES.TRIVIAL);
        expect(report.diagnosis.confidence).toBe("high");
    });

    test("rejects a series that is too short", () => {
        expect(() => analyseSeries([1, 2, 3])).toThrow(RangeError);
    });
});

describe("intercalation", () => {
    test("sqrt(2) continued fraction and convergents", () => {
        const terms = continuedFraction(Math.SQRT2, 8);
        expect(terms.slice(0, 5)).toEqual([1, 2, 2, 2, 2]);
        // sqrt(2) = [1;2,2,...] -> 1/1, 3/2, 7/5, 17/12, 41/29
        expect(convergents(terms).slice(0, 5).map((c) => [c.numerator, c.denominator])).toEqual([
            [1, 1], [3, 2], [7, 5], [17, 12], [41, 29]
        ]);
    });

    test("window lengths at convergent denominators leak least", () => {
        const ranked = rankWindowLengths(Math.SQRT2, { maxLength: 200, top: 8 });
        const best = ranked.rows.slice(0, 5).map((r) => r.length);
        expect(best.includes(70) || best.includes(169)).toBe(true);
    });

    test("the metonic cycle is 235 months in 19 years, and the 7th convergent is 4131/334", () => {
        const result = metonicAnalysis(365.2422, 29.530588, { years: 19 });
        expect(result.months).toBe(235);
        expect(result.intercalations).toBe(7);
        // the book printed 3896/315; corrected during numerical verification
        expect([result.seventhNumerator, result.seventhDenominator]).toEqual([4131, 334]);
    });
});

describe("dialogue layer", () => {
    test("detects language and extracts numbers", () => {
        expect(detectLanguage("你好")).toBe("zh");
        expect(detectLanguage("hello")).toBe("en");
        expect(extractNumbers("1, 2.5 and -3e2")).toEqual([1, 2.5, -300]);
    });

    test("routes a numeric series to the analysis tool", () => {
        const numbers = "1 2 3 4 5 6 7 8 9 10 11 12".split(" ").join(" ");
        const intent = parseIntent(numbers, []);
        expect(intent.kind).toBe("analyse");
        expect(intent.numbers).toHaveLength(12);
    });

    test("routes window questions to the intercalation tool", () => {
        expect(parseIntent("窗口取多长？", []).kind).toBe("intercalate");
        expect(parseIntent("how long should the window be", []).kind).toBe("intercalate");
    });

    test("routes concept questions to the explainer", () => {
        expect(parseIntent("什么是泄漏", []).kind).toBe("explain");
        expect(parseIntent("explain hurst", []).kind).toBe("explain");
    });

    test("the agent answers a periodicity diagnosis", async () => {
        const agent = new PeriodicityAgent({ tokenizer: tokenizer() });
        const reply = await agent.send("0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1");
        expect(reply.intent).toBe("analyse");
        expect(reply.text).toContain("平移周期");
        expect(reply.state.lastReport).not.toBeNull();
    });

    test("the agent answers a window-length question", async () => {
        const agent = new PeriodicityAgent({ tokenizer: tokenizer() });
        const reply = await agent.send("窗口取多长？周期比是 1.41421356");
        expect(reply.intent).toBe("intercalate");
        expect(reply.text).toContain("连分数");
    });

    test("the agent tokenises text into epicycle groups", async () => {
        const agent = new PeriodicityAgent({ tokenizer: tokenizer() });
        const reply = await agent.send("periodicity 周期性");
        expect(["tokenize", "explain"]).toContain(reply.intent);
        expect(reply.text.length).toBeGreaterThan(0);
    });

    test("the conversation accumulates into a waveform", async () => {
        const agent = new PeriodicityAgent({ tokenizer: tokenizer() });
        await agent.send("periodic signal");
        await agent.send("周转圆组");
        const waveform = agent.getState().waveform;
        expect(waveform).not.toBeNull();
        expect(waveform!.getEpicycles().length).toBeGreaterThan(0);
        expect(agent.topicSpectrum().length).toBeGreaterThan(0);
    });

    test("help is answered in the detected language", async () => {
        const agent = new PeriodicityAgent({ tokenizer: tokenizer() });
        const zh = await agent.send("帮助");
        expect(zh.intent).toBe("help");
        expect(zh.text).toContain("周期智能体");
    });
});
