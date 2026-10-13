import "./style.css";
import { BpeTokenizer } from "./token/bpe";
import { PeriodicityAgent, helpText } from "./dialogue/agent";
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
    void agent.send(message).then((reply) => {
        append("agent", reply.text);
        modelBadge.textContent = `${reply.intent} · 模板（本地）`;
    });
});

for (const button of Array.from(document.querySelectorAll<HTMLButtonElement>("[data-fill]"))) {
    button.addEventListener("click", () => {
        input.value = button.dataset.fill ?? "";
        input.focus();
    });
}
