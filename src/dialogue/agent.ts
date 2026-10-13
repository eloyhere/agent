import { analyseSeries, describeReport, type PeriodicityReport } from "../analysis/analyse";
import { rankWindowLengths, metonicAnalysis } from "../analysis/intercalation";
import { BpeTokenizer } from "../token/bpe";
import { tokenizeToWaveforms, type EncodedToken } from "../token/encode";
import { translate, phaseRotate, periodicConvolve, trendDiagonal } from "../mathematics/operators";
import type { Waveform } from "../mathematics/waveform";
import {
    TemplateLanguageModel,
    type ChatMessage,
    type LanguageModel
} from "./language-model";

/**
 * 可对话的周期智能体。
 *
 * 设计要点：**对话本身就是它所研究的对象**。每句话切成 token，每个 token 编码成
 * 一个 `Waveform`，整场对话则是这些波形的周期卷积累积 —— 「刚才聊了什么」对应
 * 趋势卷积的对角线（能量落在哪些频率上），「第几轮」对应 Translation 的相位旋转。
 *
 * 语言层走 `LanguageModel` 接口：现在用模板实现（离线可用），将来换 LLM 时
 * 意图识别、工具与状态都不用改。
 */

export type Language = "zh" | "en";

export type IntentKind =
    | "analyse"
    | "tokenize"
    | "intercalate"
    | "explain"
    | "help"
    | "unknown";

export interface Intent {
    readonly kind: IntentKind;
    readonly language: Language;
    readonly numbers: readonly number[];
    readonly text: string;
    readonly topic?: string;
}

export interface ConversationTurn {
    readonly role: "user" | "agent";
    readonly text: string;
    readonly intent?: IntentKind;
}

const SYSTEM_PROMPT = [
    "You are the periodicity agent built on the mathematics of 'Periodicity Mathematics'.",
    "You can: diagnose the periodicity of a numeric series (autocorrelation, periodogram,",
    "R/S Hurst, cyclic autocorrelation, and the seven-type classification of",
    "cor:data_period_type); tokenise text with byte-level BPE and encode every token as an",
    "epicycle group (ex:dft_epicycle) so that the book's operators act on it; and analyse",
    "intercalation and continued-fraction window lengths (def:intercalation_sequence_parallel).",
    "You answer only about these; you do not chat about unrelated topics."
].join(" ");

/** 判断用户用中文还是英文。 */
export const detectLanguage: (text: string) => Language = (text: string): Language =>
    /[\u4e00-\u9fff]/.test(text) ? "zh" : "en";

/** 从文本里抽出所有数字（含小数、负数、科学计数）。 */
export const extractNumbers: (text: string) => readonly number[] = (
    text: string
): readonly number[] => {
    const matches = text.match(/-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?/g);
    if (matches === null) {
        return [];
    }
    return matches.map(Number).filter((v) => Number.isFinite(v));
};

const CONCEPTS: readonly { readonly key: string; readonly zh: string[]; readonly en: string[]; readonly answer: { zh: string; en: string } }[] = [
    {
        key: "leakage",
        zh: ["泄漏", "泄露", "主瓣", "旁瓣"],
        en: ["leakage", "sidelobe", "main lobe"],
        answer: {
            zh: "泄漏＝能量落在主瓣之外的频点。书本的主瓣宽度是 4π/T（两个 bin）。注意两处已在数值验证中修正：定理 (iv) 的 η 指的是「单个峰值频点之外」而非主瓣之外（η(1/2)=1−8/π²≈0.189，不是 0.595）；而加窗频谱在频域是卷积，不是「窗核 × Dirichlet 核」的乘积。",
            en: "Leakage is energy landing on frequency points outside the main lobe. The book's main lobe is 4π/T wide (two bins). Two results were corrected in numerical verification: η in Theorem (iv) is the energy outside the SINGLE peak bin, not the main lobe (η(1/2)=1−8/π²≈0.189, not 0.595); and the windowed spectrum is a convolution in frequency, not a product of the window kernel and the Dirichlet kernel."
        }
    },
    {
        key: "epicycle",
        zh: ["周转圆", "周转", "本轮的"],
        en: ["epicycle", "fourier series"],
        answer: {
          zh: "一个周期信号可以写成周转圆之和 E(x)=Σ Aₖ·e^{i(ωₖx+φₖ)}，三要素 (A, ω, φ)。一个词也可以这样编码：把它的字节当作周期 L 的一段周期信号（词的周期就是它的长度），分解出的 (Aₖ, 2πk/L, φₖ) 就是它的周转圆组。",
          en: "A periodic signal is a sum of epicycles E(x)=Σ Aₖ·e^{i(ωₖx+φₖ)}, the triple being (A, ω, φ). A word can be encoded the same way: treat its bytes as one period of a periodic signal of length L (a word's period IS its length), and the resulting (Aₖ, 2πk/L, φₖ) are its epicycle group."
        }
    },
    {
        key: "intercalation",
        zh: ["置闰", "历法", "连分数", "窗口长度", "窗口"],
        en: ["intercalation", "calendar", "continued fraction", "window length"],
        answer: {
            zh: "置闰的量化收益来自连分数：把窗口长度取在收敛子分母 q 上，偏移 δ=‖Mρ‖ 约为 1/q_{k+1}，泄漏界 η≤(π²/3)δ² 于是从 O(1) 掉到 O(M⁻²)。窗口不是越长越好，而要落在最佳逼近的分母上。",
            en: "The quantitative payoff of intercalation comes from continued fractions: choosing the window length at a convergent denominator q makes the offset δ=‖Mρ‖ about 1/q_{k+1}, so the leakage bound η≤(π²/3)δ² drops from O(1) to O(M⁻²). A longer window is not automatically better; it has to sit on a denominator of the best approximation."
        }
    },
    {
        key: "kdi",
        zh: ["kdi", "优化器", "导数积分"],
        en: ["kdi", "optimiser", "optimizer"],
        answer: {
            zh: "KDI（核平滑导数-积分下降）是书本自创的优化器：按参数分组用不同核做加窗导数与积分估计，再合成更新。需要如实说明：书里只给了定性描述，没有给更新式；我按书的意图补全了一版，并在 54 组超参扫描下**没有**观察到它优于 Adam —— 最好的设置反而把区别性的「积分项」权重设为 0。这是阴性结果，不是证明 KDI 无用。",
            en: "KDI (kernel-smoothed derivative-integral descent) is the book's own optimiser: it estimates windowed derivatives and integrals with different kernels per parameter group and combines them. Honest caveat: the book gives only a qualitative description, no update rule. I implemented one reading of it, and over a 54-point sweep it did NOT beat Adam -- the best setting switches the distinguishing 'integral' term off. That is a negative result, not a proof that KDI is useless."
        }
    },
    {
        key: "hurst",
        zh: ["hurst", "r/s", "长程", "持续性"],
        en: ["hurst", "r/s", "persistence", "long range"],
        answer: {
            zh: "R/S Hurst 指数：H>0.5 表示长程正持续，H<0.5 反持续。但要小心：**持续不等于周期** —— 一条纯趋势也能给出 H>0.5 而完全没有任何循环。R/S 对短序列与趋势还有上偏，所以它只能作为指示，不能当结论。",
            en: "The R/S Hurst exponent: H>0.5 means long-range positive persistence, H<0.5 anti-persistence. Careful though: persistence is NOT periodicity -- a pure trend also gives H>0.5 with no cycle at all. R/S is also biased upward for short series and trends, so it is indicative only, never a verdict."
        }
    }
];

/** 识别意图。 */
export const parseIntent: (text: string, history: readonly ConversationTurn[]) => Intent = (
    text: string,
    history: readonly ConversationTurn[]
): Intent => {
    const language: Language = detectLanguage(text);
    const numbers: readonly number[] = extractNumbers(text);
    const lower: string = text.toLowerCase();

    if (/^(help|帮助|\?|？|你能做什么|what can you do)/i.test(text.trim())) {
        return { kind: "help", language, numbers, text };
    }

    // 附带足够多的数字，视为要分析一段序列
    if (numbers.length >= 8) {
        return { kind: "analyse", language, numbers, text };
    }

    // 明确的「分析上一段数据」指令
    if (/(分析|诊断|看看|解析|analyse|analyze|diagnose)/i.test(lower) && numbers.length === 0) {
        const recalled = recallDataset(history);
        if (recalled.length >= 8) {
            return { kind: "analyse", language, numbers: recalled, text };
        }
    }

    if (/(置闰|历法|连分数|窗口|intercalat|calendar|continued fraction|window)/i.test(lower)) {
        return { kind: "intercalate", language, numbers, text };
    }

    if (/(分词|token|编码|encode|转成|周转圆组)/i.test(lower)) {
        return { kind: "tokenize", language, numbers, text };
    }

    for (const concept of CONCEPTS) {
        const words = language === "zh" ? concept.zh : concept.en;
        if (words.some((w) => lower.includes(w.toLowerCase()))) {
            return { kind: "explain", language, numbers, text, topic: concept.key };
        }
    }

    // 剩下的短文本默认当作要分词的语料
    const stripped: string = text.trim();
    if (stripped.length > 0 && numbers.length < 8) {
        return { kind: "tokenize", language, numbers, text: stripped };
    }

    return { kind: "unknown", language, numbers, text };
};

/** 从对话历史里找回上一次分析过的数据集。 */
const recallDataset: (history: readonly ConversationTurn[]) => readonly number[] = (
    history: readonly ConversationTurn[]
): readonly number[] => {
    for (let i = history.length - 1; i >= 0; i -= 1) {
        const turn = history[i];
        if (turn.role !== "user") {
            continue;
        }
        const found: readonly number[] = extractNumbers(turn.text);
        if (found.length >= 8) {
            return found;
        }
    }
    return [];
};

/** 智能体的对话状态。 */
export interface AgentState {
    readonly turns: ConversationTurn[];
    /** 对话本身的波形：每句话用 token 的周转圆组经周期卷积合成。 */
    readonly waveform: Waveform | null;
    readonly lastReport: PeriodicityReport | null;
    readonly language: Language;
}

export interface AgentReply {
    readonly text: string;
    readonly intent: IntentKind;
    readonly state: AgentState;
}

/**
 * 智能体。
 *
 * 语言层走 `LanguageModel` 接口，缺省用模板实现；工具层是本项目的分析核与分词层。
 */
export class PeriodicityAgent {
    private readonly tokenizer: BpeTokenizer;
    private languageModel: LanguageModel;
    private state: AgentState;
    private readonly onWaveform: ((waveform: Waveform, tokens: readonly EncodedToken[]) => void) | null;

    public constructor(options: {
        tokenizer: BpeTokenizer;
        languageModel?: LanguageModel;
        onWaveform?: (waveform: Waveform, tokens: readonly EncodedToken[]) => void;
    }) {
        this.tokenizer = options.tokenizer;
        this.onWaveform = options.onWaveform ?? null;
        this.languageModel =
            options.languageModel ??
            new TemplateLanguageModel((context) => String(context.text ?? ""));
        this.state = { turns: [], waveform: null, lastReport: null, language: "zh" };
    }

    /**
     * 在运行期换掉语言层。
     *
     * 这是 `LanguageModel` 那道缝的用处所在：接上真实 LLM 时只调这一个方法，
     * 意图识别、工具与对话状态全都不动。
     */
    public setLanguageModel(languageModel: LanguageModel): void {
        this.languageModel = languageModel;
    }

    public getLanguageModelName(): string {
        return this.languageModel.name;
    }

    public getState(): AgentState {
        return this.state;
    }

    /** 处理一句用户输入，返回回复。 */
    public async send(message: string): Promise<AgentReply> {
        const intent: Intent = parseIntent(message, this.state.turns);
        const language: Language = intent.language;

        // 把这句话编码成波形，并累积进对话波形
        const tokens: readonly EncodedToken[] = tokenizeToWaveforms(message, this.tokenizer, {
            maxOrder: 6
        });
        const utteranceWaveforms: Waveform[] = tokens.map((t) => t.waveform);
        const accumulated: Waveform | null = this.accumulate(utteranceWaveforms);
        if (accumulated !== null && this.onWaveform !== null) {
            this.onWaveform(accumulated, tokens);
        }

        const context = this.runTool(intent);
        const text: string = await this.languageModel.complete({
            system: SYSTEM_PROMPT,
            messages: this.buildMessages(message),
            context
        });

        const turns: ConversationTurn[] = [
            ...this.state.turns,
            { role: "user", text: message, intent: intent.kind },
            { role: "agent", text }
        ];
        this.state = {
            turns,
            waveform: accumulated ?? this.state.waveform,
            lastReport: (context.report as PeriodicityReport | undefined) ?? this.state.lastReport,
            language
        };
        return { text, intent: intent.kind, state: this.state };
    }

    private buildMessages(message: string): readonly ChatMessage[] {
        const recent: ChatMessage[] = this.state.turns.slice(-6).map((t) => ({
            role: t.role === "user" ? ("user" as const) : ("assistant" as const),
            content: t.text
        }));
        return [...recent, { role: "user", content: message }];
    }

    /** 把这一句的波形并入对话波形（周期卷积）。 */
    private accumulate(utterances: readonly Waveform[]): Waveform | null {
        let combined: Waveform | null = this.state.waveform;
        for (const waveform of utterances) {
            if (combined === null) {
                combined = waveform;
            } else {
                combined = periodicConvolve(combined, waveform, {
                    useSimilarity: true,
                    d: 1,
                    keep: 16
                });
            }
        }
        return combined;
    }

    /** 调用工具，产出结构化结论。 */
    private runTool(intent: Intent): Record<string, unknown> {
        switch (intent.kind) {
            case "analyse":
                return this.toolAnalyse(intent);
            case "tokenize":
                return this.toolTokenize(intent);
            case "intercalate":
                return this.toolIntercalate(intent);
            case "explain":
                return this.toolExplain(intent);
            case "help":
                return { text: helpText(intent.language) };
            default:
                return { text: unknownText(intent.language) };
        }
    }

    private toolAnalyse(intent: Intent): Record<string, unknown> {
        const values = Float64Array.from(intent.numbers);
        const report: PeriodicityReport = analyseSeries(values);
        const body: string = describeReport(report);
        const header: string =
            intent.language === "zh"
                ? `对这段 ${values.length} 点的序列做周期性诊断：`
                : `Periodicity diagnosis of your ${values.length}-point series:`;
        const tail: string =
            intent.language === "zh"
                ? "\n\n要追问某个周期、或想换一段数据，直接说。"
                : "\n\nAsk about any of these periods, or paste another series.";
        return { text: `${header}\n${body}${tail}`, report };
    }

    private toolTokenize(intent: Intent): Record<string, unknown> {
        const tokens: readonly EncodedToken[] = tokenizeToWaveforms(intent.text, this.tokenizer, {
            maxOrder: 6
        });
        const byteLength: number = new TextEncoder().encode(intent.text).length;
        const lines: string[] = [];
        lines.push(
            intent.language === "zh"
                ? `「${intent.text}」→ ${tokens.length} 个 token（${byteLength} 字节，${(byteLength / Math.max(1, tokens.length)).toFixed(2)} 字节/token）`
                : `"${intent.text}" -> ${tokens.length} tokens (${byteLength} bytes, ${(byteLength / Math.max(1, tokens.length)).toFixed(2)} bytes/token)`
        );
        lines.push(intent.language === "zh" ? "\ntoken → 周转圆组：" : "\ntoken -> epicycle group:");
        for (const token of tokens.slice(0, 8)) {
            if (token.period < 2) {
                continue;
            }
            const top = token.waveform.getEpicycles().slice().sort((a, b) => b.getAmplitude() - a.getAmplitude())[0];
            if (top === undefined) {
                continue;
            }
            lines.push(
                `  ${JSON.stringify(token.text).padEnd(12)} 周期=${String(token.period).padStart(2)}  ` +
                    `周转圆=${String(token.waveform.getEpicycles().length).padStart(2)}  ` +
                    `最强 k=${String(Math.round((top.getOmega() * token.period) / (2 * Math.PI))).padStart(2)} ` +
                    `A=${top.getAmplitude().toFixed(3)} φ=${top.getPhase().toFixed(3)}`
            );
        }
        return { text: lines.join("\n") };
    }

    private toolIntercalate(intent: Intent): Record<string, unknown> {
        // 两个数且第一个明显大于第二个 → 当作年长 / 月长
        if (intent.numbers.length >= 2 && intent.numbers[0] > intent.numbers[1]) {
            const result = metonicAnalysis(intent.numbers[0], intent.numbers[1], { years: 19 });
            const lines: string[] =
                intent.language === "zh"
                    ? [
                          `年长 ${intent.numbers[0]}、月长 ${intent.numbers[1]}：`,
                          `周期比 ρ ≈ ${result.rho.toFixed(6)}`,
                          `连分数 [${result.terms.slice(0, 7).join("; ")}…]`,
                          `十九年得 ${result.months} 个月，即 ${result.intercalations} 次置闰`,
                          `偏移 δ = ${result.delta.toExponential(3)}`,
                          `泄漏界 η ≤ (π²/3)δ² ≈ ${result.eta.toExponential(3)}`,
                          `第七个收敛子为 ${result.seventhNumerator}/${result.seventhDenominator}（书本原写 3896/315，验证中已修正为 4131/334）`
                      ]
                    : [
                          `Year ${intent.numbers[0]}, month ${intent.numbers[1]}:`,
                          `ratio rho ~ ${result.rho.toFixed(6)}`,
                          `continued fraction [${result.terms.slice(0, 7).join("; ")}...]`,
                          `19 years hold ${result.months} months, i.e. ${result.intercalations} intercalations`,
                          `offset delta = ${result.delta.toExponential(3)}`,
                          `leakage bound eta <= (pi^2/3)delta^2 ~ ${result.eta.toExponential(3)}`,
                          `the seventh convergent is ${result.seventhNumerator}/${result.seventhDenominator} (the book printed 3896/315; corrected to 4131/334 during verification)`
                      ];
            return { text: lines.join("\n") };
        }
        const rho: number = intent.numbers.length >= 1 ? intent.numbers[0] : Math.SQRT2;
        const ranked = rankWindowLengths(rho, { maxLength: 200, top: 8 });
        const lines: string[] = [];
        lines.push(
            intent.language === "zh"
                ? `对周期比 ρ = ${rho}：连分数 [${ranked.terms.slice(0, 6).join("; ")}…]`
                : `For rho = ${rho}: continued fraction [${ranked.terms.slice(0, 6).join("; ")}...]`
        );
        lines.push(intent.language === "zh" ? "最省泄漏的窗口长度：" : "Window lengths with the least leakage:");
        for (const row of ranked.rows.slice(0, 5)) {
            lines.push(
                `  M=${String(row.length).padStart(3)}  δ=${row.delta.toFixed(5)}  η≈${row.eta.toExponential(3)}` +
                    (row.isConvergentDenominator ? (intent.language === "zh" ? "  ← 连分数分母" : "  <- convergent") : "")
            );
        }
        lines.push(
            intent.language === "zh"
                ? "\n结论：窗口不是越长越好，而要落在最佳逼近的分母上。"
                : "\nConclusion: a longer window is not better; it has to sit on a denominator of the best approximation."
        );
        return { text: lines.join("\n") };
    }

    private toolExplain(intent: Intent): Record<string, unknown> {
        const concept = CONCEPTS.find((c) => c.key === intent.topic);
        if (concept === undefined) {
            return { text: unknownText(intent.language) };
        }
        return { text: concept.answer[intent.language] };
    }

    /** 对话波形的能量分布，用于「刚才聊了什么」。 */
    public topicSpectrum(): readonly number[] {
        const waveform: Waveform | null = this.state.waveform;
        if (waveform === null) {
            return [];
        }
        return [...trendDiagonal(waveform)].sort((a, b) => b - a).slice(0, 8);
    }

    /** 演示：对对话波形作一次平移与旋转，验证算子可用。 */
    public transformWaveform(tau: number, theta: number): Waveform | null {
        const waveform: Waveform | null = this.state.waveform;
        if (waveform === null) {
            return null;
        }
        return phaseRotate(translate(waveform, tau), theta);
    }
}

export const helpText = (language: Language): string =>
    language === "zh"
        ? [
              "我是基于《Periodicity Mathematics》的周期智能体。可以：",
              "  1. 周期性诊断 —— 贴一段数字（≥8 个，逗号或空格分隔），我给自相关、周期图、R/S、循环自相关与七分类判定",
              "  2. 分词与周转圆组 —— 给我一段文字，我切成 BPE token 并把每个 token 编码成周转圆组",
              "  3. 置闰与窗口 —— 问「窗口取多长」，或给我年长与月长两个数，我算连分数与泄漏界",
              "  4. 概念解释 —— 问泄漏、周转圆、Hurst、KDI 等",
              "边界说明：我只能聊这些，不做闲聊或与周期无关的问答。"
          ].join("\n")
        : [
              "I am the periodicity agent built on 'Periodicity Mathematics'. I can:",
              "  1. Diagnose periodicity -- paste a numeric series (>= 8 values). I report autocorrelation, periodogram, R/S, cyclic autocorrelation and the seven-type classification.",
              "  2. Tokenise -- give me text; I split it with byte-level BPE and encode each token as an epicycle group.",
              "  3. Intercalation -- ask how long a window should be, or give me a year length and a month length.",
              "  4. Explain concepts -- leakage, epicycles, Hurst, KDI.",
              "Boundary: I only talk about these. No small talk, no unrelated questions."
          ].join("\n");

export const unknownText = (language: Language): string =>
    language === "zh"
        ? "没听明白。可以试试：贴一段数字让我做周期性诊断、给我一段文字让我分词并编码成周转圆组、或问「窗口取多长」。输入「帮助」看全部能力。"
        : "I did not follow that. Try pasting a numeric series for a periodicity diagnosis, giving me text to tokenise into epicycle groups, or asking how long a window should be. Type 'help' for the full list.";
