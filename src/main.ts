import "./style.css";
import { BpeTokenizer } from "./token/bpe";
import { PeriodicityAgent, helpText } from "./dialogue/agent";
import { TemplateLanguageModel, HttpLanguageModel } from "./dialogue/language-model";
import type { EncodedToken } from "./token/encode";
import type { Waveform } from "./mathematics/waveform";

/**
 * 聊天界面。
 *
 * 启动时用一段小语料训练字节级 BPE（浏览器里就能跑完），随后用户输入经由
 * `PeriodicityAgent` 处理：意图识别 → 调用书本工具 → 语言层渲染成回复。
 *
 * 右侧面板显示**对话波形**的能量分布 —— 因为整场对话本身就是一个 `Waveform`，
 * 「刚才聊了什么」对应趋势卷积对角线上的能量分布。
 */

const SEED_CORPUS = [
    "periodicity periodic signal signals sampling sample window leakage",
    "epicycle epicycles waveform amplitude omega phase frequency harmonic",
    "intercalation calendar continued fraction convergent denominator offset",
    "hurst persistence autocorrelation periodogram spectrum peak bin sidelobe",
    "the quick brown fox jumps over the lazy dog",
    "周期性 周期 信号 采样 窗口 泄漏 周转圆 振幅 频率 相位 谐波",
    "置闰 历法 连分数 收敛子 分母 偏移 自相关 周期图 谱 峰",
    "分词 编码 智能体 对话 分析 诊断 判定 周期"
].join("\n");

const root = document.querySelector<HTMLDivElement>("#app");
if (root === null) {
    throw new Error("#app not found");
}
root.innerHTML = `
<main class="shell">
  <header class="bar">
    <h1>periodicity agent</h1>
    <span class="sub">基于《Periodicity Mathematics》· 本地引擎 · 零依赖</span>
    <span id="model" class="badge"></span>
  </header>
  <section class="split">
    <div class="chat">
      <div id="log" class="log"></div>
      <form id="composer" class="composer">
        <textarea id="input" rows="2" placeholder="贴一段数字做周期性诊断；或给我一段文字做分词与周转圆编码；或问「窗口取多长」。输入「帮助」看全部能力。"></textarea>
        <button type="submit">发送</button>
      </form>
      <div class="hints">
        <button data-fill="帮助" type="button">帮助</button>
        <button data-fill="窗口取多长？周期比是 1.41421356" type="button">窗口长度</button>
        <button data-fill="年长 365.2422 月长 29.530588" type="button">十九年七闰</button>
        <button data-fill="0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1" type="button">纯周期序列</button>
        <button data-fill="0 1 4 9 16 25 36 49 64 81 100 121 144" type="button">趋势序列</button>
        <button data-fill="周期性 与 周转圆组" type="button">分词示例</button>
      </div>
    </div>
    <aside class="side">
      <h2>对话波形</h2>
      <p class="note">整场对话累积成一个 <code>Waveform</code>。下列是趋势卷积对角线上的能量 —— 即「刚才聊的是什么」在频率上的写照。</p>
      <div id="spectrum" class="spectrum"></div>
      <h2>分词器</h2>
      <div id="tokenizer" class="kv"></div>
      <h2>语言层</h2>
      <p class="note">缺省是本地模板引擎（离线、零依赖）。填入任意 OpenAI 兼容端点即可切到真实模型 —— 意图识别、工具与对话状态都不会变。</p>
      <div class="llm">
        <label>端点 <input id="llm-endpoint" type="text" placeholder="https://api.example.com/v1/chat/completions"></label>
        <label>模型 <input id="llm-model" type="text" placeholder="deepseek-chat"></label>
        <label>密钥 <input id="llm-key" type="password" placeholder="sk-..."></label>
        <div class="llm-actions">
          <button id="llm-apply" type="button">启用</button>
          <button id="llm-reset" type="button">回到本地</button>
        </div>
        <p id="llm-status" class="note"></p>
      </div>
    </aside>
  </section>
</main>`;

const log = document.querySelector<HTMLDivElement>("#log")!;
const form = document.querySelector<HTMLFormElement>("#composer")!;
const input = document.querySelector<HTMLTextAreaElement>("#input")!;
const spectrumBox = document.querySelector<HTMLDivElement>("#spectrum")!;
const tokenizerBox = document.querySelector<HTMLDivElement>("#tokenizer")!;
const modelBadge = document.querySelector<HTMLSpanElement>("#model")!;

/** 追加一条消息。 */
const append = (role: "user" | "agent", text: string): void => {
    const element = document.createElement("div");
    element.className = `msg ${role}`;
    const paragraph = document.createElement("pre");
    paragraph.textContent = text;
    element.appendChild(paragraph);
    log.appendChild(element);
    log.scrollTop = log.scrollHeight;
};

/** 渲染对话波形的能量分布。 */
const renderSpectrum = (waveform: Waveform, tokens: readonly EncodedToken[]): void => {
    const epicycles = waveform.getEpicycles();
    const total = epicycles.reduce((sum, e) => sum + e.getAmplitude() ** 2, 0);
    if (total <= 0) {
        spectrumBox.innerHTML = '<p class="note">还没有内容。</p>';
        return;
    }
    const rows = epicycles
        .map((e) => ({ omega: e.getOmega(), energy: e.getAmplitude() ** 2 }))
        .sort((a, b) => b.energy - a.energy)
        .slice(0, 10);
    spectrumBox.innerHTML = rows
        .map((row) => {
            const share = row.energy / total;
            return `<div class="row"><span class="label">ω=${row.omega.toFixed(3)}</span>
        <span class="meter"><i style="width:${(share * 100).toFixed(1)}%"></i></span>
        <span class="value">${(share * 100).toFixed(1)}%</span></div>`;
        })
        .join("");
    void tokens;
};

/** 训练分词器并显示统计。 */
const tokenizer = new BpeTokenizer();
tokenizer.train(SEED_CORPUS, { vocabularySize: 420, maxCorpusBytes: 4000 });
const tokenizerStats = tokenizer.stats();
tokenizerBox.innerHTML = [
    `<div class="row"><span class="label">合并数</span><span class="value">${tokenizerStats.merges}</span></div>`,
    `<div class="row"><span class="label">词表</span><span class="value">${tokenizerStats.vocabularySize}</span></div>`,
    `<div class="row"><span class="label">最长 token</span><span class="value">${tokenizerStats.longestTokenBytes} 字节</span></div>`
].join("");

const agent = new PeriodicityAgent({
    tokenizer,
    onWaveform: (waveform) => renderSpectrum(waveform, [])
});

modelBadge.textContent = "语言层：模板（本地）";
append("agent", helpText("zh"));

form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (message.length === 0) {
        return;
    }
    input.value = "";
    append("user", message);
    void agent
        .send(message)
        .then((reply) => {
            append("agent", reply.text);
            modelBadge.textContent = `${reply.intent} · ${agent.getLanguageModelName()}`;
        })
        .catch((error: unknown) => {
            // 常见失败是浏览器被 CORS 拦下；此时不要静默，直接说清并提示退回本地
            const detail = error instanceof Error ? error.message : String(error);
            append("agent", `语言层调用失败：${detail}\n\n若是 CORS 相关错误，浏览器直连第三方 API 会被拦下，需要同源转发端点；也可以点右侧「回到本地」继续用模板引擎。`);
        });
});

for (const button of Array.from(document.querySelectorAll<HTMLButtonElement>("[data-fill]"))) {
    button.addEventListener("click", () => {
        input.value = button.dataset.fill ?? "";
        input.focus();
    });
}

// ---------------------------------------------------------------- 语言层切换
// 用 localStorage 记住设置，刷新后仍然生效。密钥只存在本机浏览器里，不发给任何第三方。
const LLM_STORAGE_KEY = "periodicity-agent.language-model";
const endpointInput = document.querySelector<HTMLInputElement>("#llm-endpoint")!;
const modelInput = document.querySelector<HTMLInputElement>("#llm-model")!;
const keyInput = document.querySelector<HTMLInputElement>("#llm-key")!;
const statusLine = document.querySelector<HTMLParagraphElement>("#llm-status")!;
const localModel = new TemplateLanguageModel((context) => String(context.text ?? ""));

const setStatus = (text: string): void => {
    statusLine.textContent = text;
};

const applyLocal = (): void => {
    agent.setLanguageModel(localModel);
    modelBadge.textContent = "模板（本地）";
    setStatus("当前：本地模板引擎。");
};

const saved = localStorage.getItem(LLM_STORAGE_KEY);
if (saved !== null) {
    try {
        const parsed = JSON.parse(saved) as { endpoint?: string; model?: string; key?: string };
        endpointInput.value = parsed.endpoint ?? "";
        modelInput.value = parsed.model ?? "";
        keyInput.value = parsed.key ?? "";
        if (parsed.endpoint && parsed.model && parsed.key) {
            agent.setLanguageModel(
                new HttpLanguageModel({
                    endpoint: parsed.endpoint,
                    model: parsed.model,
                    apiKey: parsed.key
                })
            );
            modelBadge.textContent = `HTTP · ${parsed.model}`;
            setStatus(`当前：${parsed.model}`);
        } else {
            setStatus("当前：本地模板引擎。");
        }
    } catch {
        setStatus("当前：本地模板引擎（已存的设置无法解析，已忽略）。");
    }
} else {
    setStatus("当前：本地模板引擎。");
}

document.querySelector<HTMLButtonElement>("#llm-apply")!.addEventListener("click", () => {
    const endpoint = endpointInput.value.trim();
    const model = modelInput.value.trim();
    const key = keyInput.value.trim();
    if (!endpoint || !model || !key) {
        setStatus("端点、模型、密钥三项都要填。");
        return;
    }
    localStorage.setItem(LLM_STORAGE_KEY, JSON.stringify({ endpoint, model, key }));
    agent.setLanguageModel(new HttpLanguageModel({ endpoint, model, apiKey: key }));
    modelBadge.textContent = `HTTP · ${model}`;
    setStatus(`已切到 ${model}。若发送失败，多半是浏览器被 CORS 拦下，需要一个同源转发端点。`);
});

document.querySelector<HTMLButtonElement>("#llm-reset")!.addEventListener("click", () => {
    localStorage.removeItem(LLM_STORAGE_KEY);
    applyLocal();
});
