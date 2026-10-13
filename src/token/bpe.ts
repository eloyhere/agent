/**
 * 字节级 BPE 分词器。
 *
 * 字节级（而非字符级）意味着**不存在词表外输入**：任何字符串都是字节序列，
 * 基础词表就是 256 个字节值。学到的合并按秩排序，编码时从低秩开始套用，
 * 这保证 `decode(encode(s)) === s` 逐字节成立。
 *
 * 与书本的联系：一个 token 随后会被编码成**周转圆组**（见 encode.ts），
 * 于是书本的算子可以直接作用在 token 上。
 */

const BASE = 256;

export interface BpeData {
    readonly merges: readonly (readonly [number, number])[];
    readonly specials: Readonly<Record<string, number>>;
}

export class BpeTokenizer {
    private readonly mergeList: [number, number][];
    private readonly specialMap: Record<string, number>;
    private rank: Map<string, number>;
    private mergedId: Map<string, number>;

    public constructor(data: BpeData = { merges: [], specials: {} }) {
        this.mergeList = data.merges.map((m) => [m[0], m[1]]);
        this.specialMap = { ...data.specials };
        this.rank = new Map();
        this.mergedId = new Map();
        this.reindex();
    }

    private reindex(): void {
        this.rank = new Map();
        this.mergedId = new Map();
        for (let i = 0; i < this.mergeList.length; i += 1) {
            const key = `${this.mergeList[i][0]},${this.mergeList[i][1]}`;
            this.rank.set(key, i);
            this.mergedId.set(key, BASE + i);
        }
    }

    public get merges(): readonly (readonly [number, number])[] {
        return this.mergeList;
    }

    public getVocabularySize(): number {
        return BASE + this.mergeList.length + Object.keys(this.specialMap).length;
    }

    /** token id 对应的字节序列。 */
    public tokenBytes(id: number): Uint8Array {
        if (id < BASE) {
            return Uint8Array.of(id);
        }
        if (id < BASE + this.mergeList.length) {
            const [a, b] = this.mergeList[id - BASE];
            const left = this.tokenBytes(a);
            const right = this.tokenBytes(b);
            const out = new Uint8Array(left.length + right.length);
            out.set(left, 0);
            out.set(right, left.length);
            return out;
        }
        return new Uint8Array(0);
    }

    /** token id 对应的文本（部分 UTF-8 会有损）。 */
    public tokenText(id: number): string {
        return new TextDecoder().decode(this.tokenBytes(id));
    }

    /**
     * 从语料学习合并。
     *
     * `maxCorpusBytes` 截断语料以保证训练速度；截断点会退到多字节字符边界，
     * 不会把字符劈成两半。
     */
    public train(
        corpus: string,
        options: { vocabularySize?: number; maxCorpusBytes?: number; specials?: readonly string[] } = {}
    ): this {
        const vocabularySize = options.vocabularySize ?? 1024;
        const maxCorpusBytes = options.maxCorpusBytes ?? 300000;
        if (vocabularySize < BASE) {
            throw new RangeError("vocabularySize must be at least 256 for byte-level BPE");
        }

        let bytes = new TextEncoder().encode(corpus);
        if (bytes.length > maxCorpusBytes) {
            bytes = bytes.subarray(0, maxCorpusBytes);
            while (bytes.length > 0 && (bytes[bytes.length - 1] & 0xc0) === 0x80) {
                bytes = bytes.subarray(0, bytes.length - 1);
            }
        }

        // 按空白切成「词」，合并不跨空白 —— 常规做法，也让解码精确
        const words: number[][] = [];
        let current: number[] = [];
        for (const byte of bytes) {
            if (byte === 0x20 || byte === 0x0a || byte === 0x09 || byte === 0x0d) {
                if (current.length > 0) {
                    words.push(current);
                    current = [];
                }
                words.push([byte]);
            } else {
                current.push(byte);
            }
        }
        if (current.length > 0) {
            words.push(current);
        }

        // 按「词型」计数，合并是在词型上学的，不是按出现次数加权到每个词上
        const counts = new Map<string, number>();
        for (const word of words) {
            const key = word.join(",");
            counts.set(key, (counts.get(key) ?? 0) + 1);
        }
        let sequences: number[][] = [];
        let frequencies: number[] = [];
        for (const [key, frequency] of counts) {
            sequences.push(key.split(",").map(Number));
            frequencies.push(frequency);
        }

        const targets = vocabularySize - BASE;
        for (let step = 0; step < targets; step += 1) {
            const pairCounts = new Map<string, number>();
            for (let i = 0; i < sequences.length; i += 1) {
                const sequence = sequences[i];
                const frequency = frequencies[i];
                for (let j = 0; j + 1 < sequence.length; j += 1) {
                    const key = `${sequence[j]},${sequence[j + 1]}`;
                    pairCounts.set(key, (pairCounts.get(key) ?? 0) + frequency);
                }
            }
            if (pairCounts.size === 0) {
                break;
            }
            let bestKey: string | null = null;
            let bestCount = 0;
            for (const [key, count] of pairCounts) {
                // 平局按键名决胜，保证训练可复现
                if (count > bestCount || (count === bestCount && bestKey !== null && key < bestKey)) {
                    bestCount = count;
                    bestKey = key;
                }
            }
            if (bestKey === null || bestCount < 2) {
                break;
            }
            const [a, b] = bestKey.split(",").map(Number);
            this.mergeList.push([a, b]);
            const newId = BASE + this.mergeList.length - 1;

            for (let i = 0; i < sequences.length; i += 1) {
                const sequence = sequences[i];
                if (sequence.length < 2) {
                    continue;
                }
                const out: number[] = [];
                for (let j = 0; j < sequence.length; j += 1) {
                    if (j + 1 < sequence.length && sequence[j] === a && sequence[j + 1] === b) {
                        out.push(newId);
                        j += 1;
                    } else {
                        out.push(sequence[j]);
                    }
                }
                sequences[i] = out;
            }
            void step;
        }

        for (const name of options.specials ?? []) {
            this.specialMap[name] = BASE + this.mergeList.length + Object.keys(this.specialMap).length;
        }
        this.reindex();
        return this;
    }

    /** 编码为 token id 序列。 */
    public encode(text: string): number[] {
        const out: number[] = [];
        const names = Object.keys(this.specialMap).sort((a, b) => b.length - a.length);
        const chunks: string[] =
            names.length > 0
                ? text.split(new RegExp(`(${names.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "g"))
                : [text];

        for (const chunk of chunks) {
            if (this.specialMap[chunk] !== undefined) {
                out.push(this.specialMap[chunk]);
                continue;
            }
            if (chunk.length === 0) {
                continue;
            }
            const bytes = new TextEncoder().encode(chunk);
            let current: number[] = [];
            for (const byte of bytes) {
                if (byte === 0x20 || byte === 0x0a || byte === 0x09 || byte === 0x0d) {
                    if (current.length > 0) {
                        out.push(...this.encodeWord(current));
                        current = [];
                    }
                    out.push(...this.encodeWord([byte]));
                } else {
                    current.push(byte);
                }
            }
            if (current.length > 0) {
                out.push(...this.encodeWord(current));
            }
        }
        return out;
    }

    /** 对一个空白分隔的字节词套用已学到的合并。 */
    private encodeWord(bytes: readonly number[]): number[] {
        let sequence: number[] = [...bytes];
        for (;;) {
            let bestRank = Number.POSITIVE_INFINITY;
            let bestPair: string | null = null;
            for (let j = 0; j + 1 < sequence.length; j += 1) {
                const key = `${sequence[j]},${sequence[j + 1]}`;
                const rank = this.rank.get(key);
                if (rank !== undefined && rank < bestRank) {
                    bestRank = rank;
                    bestPair = key;
                }
            }
            if (bestPair === null) {
                break;
            }
            const newId = BASE + bestRank;
            const out: number[] = [];
            for (let j = 0; j < sequence.length; j += 1) {
                if (j + 1 < sequence.length && `${sequence[j]},${sequence[j + 1]}` === bestPair) {
                    out.push(newId);
                    j += 1;
                } else {
                    out.push(sequence[j]);
                }
            }
            sequence = out;
        }
        return sequence;
    }

    /** 解码回文本。 */
    public decode(ids: readonly number[]): string {
        const parts: Uint8Array[] = [];
        for (const id of ids) {
            if (id < BASE + this.mergeList.length) {
                parts.push(this.tokenBytes(id));
            } else {
                for (const [name, sid] of Object.entries(this.specialMap)) {
                    if (sid === id) {
                        parts.push(new TextEncoder().encode(name));
                        break;
                    }
                }
            }
        }
        let length = 0;
        for (const part of parts) {
            length += part.length;
        }
        const merged = new Uint8Array(length);
        let offset = 0;
        for (const part of parts) {
            merged.set(part, offset);
            offset += part.length;
        }
        return new TextDecoder().decode(merged);
    }

    /** 序列化。 */
    public save(): BpeData {
        return { merges: this.mergeList.map((m) => [m[0], m[1]] as const), specials: this.specialMap };
    }

    public static load(data: BpeData): BpeTokenizer {
        return new BpeTokenizer(data);
    }

    public stats(): { merges: number; vocabularySize: number; longestTokenBytes: number } {
        let longest = 1;
        for (let i = 0; i < this.mergeList.length; i += 1) {
            longest = Math.max(longest, this.tokenBytes(BASE + i).length);
        }
        return {
            merges: this.mergeList.length,
            vocabularySize: this.getVocabularySize(),
            longestTokenBytes: longest
        };
    }
}
