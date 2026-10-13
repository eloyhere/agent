/**
 * 语言模型接口 —— 本地引擎与将来接 LLM 之间的那道缝。
 *
 * 智能体的对话层只依赖这个接口，不依赖任何具体实现。于是：
 *   * 现在用 `TemplateLanguageModel`：零依赖、离线、即时，语言由模板产出；
 *   * 将来接上任何 LLM，只需另写一个实现，对话层与工具层一行不改。
 */

export interface ChatMessage {
    readonly role: "system" | "user" | "assistant";
    readonly content: string;
}

export interface CompletionRequest {
    /** 系统提示，描述智能体的身份与可用工具。 */
    readonly system: string;
    /** 对话历史，最后一条是当前用户输入。 */
    readonly messages: readonly ChatMessage[];
    /** 工具层的结构化结果，模板模型据此组织措辞。 */
    readonly context?: Readonly<Record<string, unknown>>;
}

export interface LanguageModel {
    readonly name: string;
    /** 是否为本地的、离线的实现。 */
    readonly isLocal: boolean;
    complete(request: CompletionRequest): Promise<string>;
}

/**
 * 本地模板模型：不做自由生成，只把工具结果组织成自然语言。
 *
 * 必须说清的边界：它能就**周期性分析、分词、置闰、书本定理**这些话题对答，
 * 也能引用上文；但**不会**闲聊、写诗或回答与周期无关的问题。`complete` 因此
 * 收到 `context` 里已经算好的结论，而不是自己推理。
 */
export class TemplateLanguageModel implements LanguageModel {
    public readonly name: string = "template";
    public readonly isLocal: boolean = true;

    /** 由对话层注入的渲染函数：把结构化结论变成文字。 */
    private readonly render: (context: Readonly<Record<string, unknown>>) => string;

    public constructor(render: (context: Readonly<Record<string, unknown>>) => string) {
        this.render = render;
    }

    public async complete(request: CompletionRequest): Promise<string> {
        return this.render(request.context ?? {});
    }
}

/**
 * 一个可选的 HTTP 实现骨架，供接入真实 LLM。
 *
 * 有意**不**在此处硬编码任何厂商：端点、模型名与密钥都由调用方传入。
 * 注意浏览器直连第三方 API 通常会被 CORS 拦下，届时需要一个同源转发端点。
 */
export class HttpLanguageModel implements LanguageModel {
    public readonly name: string;
    public readonly isLocal: boolean = false;
    private readonly endpoint: string;
    private readonly apiKey: string;
    private readonly model: string;
    private readonly fetchImpl: typeof fetch;

    public constructor(options: {
        endpoint: string;
        apiKey: string;
        model: string;
        name?: string;
        fetchImpl?: typeof fetch;
    }) {
        this.endpoint = options.endpoint;
        this.apiKey = options.apiKey;
        this.model = options.model;
        this.name = options.name ?? "http";
        this.fetchImpl = options.fetchImpl ?? fetch;
    }

    public async complete(request: CompletionRequest): Promise<string> {
        const response = await this.fetchImpl(this.endpoint, {
            method: "POST",
            headers: {
                "content-type": "application/json",
                authorization: `Bearer ${this.apiKey}`
            },
            body: JSON.stringify({
                model: this.model,
                messages: [
                    { role: "system", content: request.system },
                    ...request.messages
                ]
            })
        });
        if (!response.ok) {
            throw new Error(`language model request failed: ${response.status}`);
        }
        const payload = (await response.json()) as {
            choices?: { message?: { content?: string } }[];
        };
        const text: string | undefined = payload.choices?.[0]?.message?.content;
        if (typeof text !== "string") {
            throw new Error("language model returned no text");
        }
        return text;
    }
}
