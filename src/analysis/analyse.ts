import {
    autocorrelation,
    autocorrelationPeaks,
    hurstRS,
    cyclicAutocorrelation,
    mean,
    variance,
    type AutocorrelationPeak,
    type CyclicScan
} from "./periods";
import { periodogram, findPeaks, totalPower, bandPowerShare, type RefinedPeak } from "./spectrum";

/**
 * 周期性判定流水线 —— 书本第 31 章的六个诊断步骤。
 *
 * 平移周期（自相关）、旋转周期（周期图）、缩放周期（R/S）、循环自相关，
 * 再按 `cor:data_period_type` 的七分类给出类型。
 *
 * 分类是**刻意保守**的：只用流水线真正产出的证据，证据不足以区分类型时
 * 直接报 `undetermined`。数值证据能**否证**一个类型，不能**证明**一个。
 */

export const PERIOD_TYPES: {
    readonly TRIVIAL: "trivially-periodic";
    readonly STRICT: "strictly-periodic";
    readonly UNIFORM: "uniformly-periodic";
    readonly QUASI: "quasi-periodic";
    readonly EVENTUALLY: "eventually-periodic";
    readonly DISCRETE: "discretely-periodic";
    readonly DAMPED: "asymptotically-periodic";
    readonly UNDETERMINED: "undetermined";
} = {
    TRIVIAL: "trivially-periodic",
    STRICT: "strictly-periodic",
    UNIFORM: "uniformly-periodic",
    QUASI: "quasi-periodic",
    EVENTUALLY: "eventually-periodic",
    DISCRETE: "discretely-periodic",
    DAMPED: "asymptotically-periodic",
    UNDETERMINED: "undetermined"
} as const;

export type PeriodType = (typeof PERIOD_TYPES)[keyof typeof PERIOD_TYPES];

export interface Diagnosis {
    readonly type: PeriodType;
    readonly confidence: "high" | "medium" | "low" | "none";
    readonly reasons: readonly string[];
    readonly caveats: readonly string[];
}

export interface PeriodicityReport {
    readonly length: number;
    readonly mean: number;
    readonly standardDeviation: number;
    readonly autocorrelationPeaks: readonly AutocorrelationPeak[];
    readonly peaks: readonly RefinedPeak[];
    readonly topPeakShare: number;
    readonly hurst: number;
    readonly cyclic: CyclicScan;
    readonly diagnosis: Diagnosis;
}

export interface AnalyseOptions {
    readonly maxLag?: number | null;
    readonly topPeaks?: number;
    readonly detrend?: boolean;
    /** R/S 的最小分段长度；缺省由序列长度推出。 */
    readonly minChunk?: number;
}

export const analyseSeries: (values: ArrayLike<number>, options?: AnalyseOptions) => PeriodicityReport = (
    values: ArrayLike<number>,
    options: AnalyseOptions = {}
): PeriodicityReport => {
    const n: number = values.length;
    if (n < 8) {
        throw new RangeError("analyseSeries: need at least 8 samples");
    }
    const topPeaks: number = options.topPeaks ?? 6;
    const detrend: boolean = options.detrend ?? true;
    const maxLag: number =
        options.maxLag ?? Math.min(Math.floor(n / 2), Math.max(32, Math.floor(4 * Math.sqrt(n))));

    const autocorrelationValues: Float64Array = autocorrelation(values, { maxLag, detrend });
    const acPeaks: readonly AutocorrelationPeak[] = autocorrelationPeaks(autocorrelationValues, {
        minLag: 2,
        count: 6
    });

    const spectrum = periodogram(values, { detrend });
    const peaks: readonly RefinedPeak[] = findPeaks(spectrum.frequencies, spectrum.power, {
        count: topPeaks
    });
    const total: number = totalPower(spectrum.power);
    const topPeakShare: number = peaks.length > 0 && total > 0 ? peaks[0].power / total : 0;

    const minChunk: number = options.minChunk ?? Math.max(8, Math.floor(Math.sqrt(n) / 2));
    const hurst: number = hurstRS(values, { minChunk });

    const cyclic: CyclicScan = cyclicAutocorrelation(values, { alphaMax: 0.25, count: 2000 });

    const m: number = mean(values);
    const sd: number = Math.sqrt(variance(values));

    const partial: Omit<PeriodicityReport, "diagnosis"> = {
        length: n,
        mean: m,
        standardDeviation: sd,
        autocorrelationPeaks: acPeaks,
        peaks,
        topPeakShare,
        hurst,
        cyclic
    };
    return { ...partial, diagnosis: classifyPeriodicity(partial) };
};

/** 按 `cor:data_period_type` 分类。 */
export const classifyPeriodicity: (
    report: Omit<PeriodicityReport, "diagnosis"> | PeriodicityReport
) => Diagnosis = (report: Omit<PeriodicityReport, "diagnosis"> | PeriodicityReport): Diagnosis => {
    const reasons: string[] = [];
    const caveats: string[] = [];

    if (report.standardDeviation === 0) {
        return {
            type: PERIOD_TYPES.TRIVIAL,
            confidence: "high",
            reasons: ["序列为常数，方差为零"],
            caveats
        };
    }
    if (report.peaks.length === 0) {
        return {
            type: PERIOD_TYPES.UNDETERMINED,
            confidence: "none",
            reasons: ["谱中找不到任何峰"],
            caveats
        };
    }

    const top: RefinedPeak = report.peaks[0];
    reasons.push(
        `主导周期约 ${top.period.toFixed(2)} 个采样，占周期图功率的 ${(report.topPeakShare * 100).toFixed(1)}%`
    );

    const secondary: readonly RefinedPeak[] = report.peaks
        .slice(1)
        .filter((p) => p.power > 0.1 * top.power);
    const strongCount: number = 1 + secondary.length;

    const acPeak: AutocorrelationPeak | undefined = report.autocorrelationPeaks[0];
    const acMatches: boolean =
        acPeak !== undefined && Math.abs(acPeak.lag - top.period) / top.period < 0.25;

    if (report.topPeakShare > 0.5 && strongCount <= 2 && acMatches) {
        reasons.push(`自相关在滞后 ${acPeak.lag} 处出现局部极大，与谱峰周期吻合`);
        return { type: PERIOD_TYPES.STRICT, confidence: "high", reasons, caveats };
    }

    if (strongCount >= 3) {
        const periods: number[] = [top, ...secondary].map((p) => p.period).sort((a, b) => a - b);
        let commensurable = true;
        for (let i = 1; i < periods.length; i += 1) {
            const ratio: number = periods[i] / periods[0];
            if (Math.abs(ratio - Math.round(ratio)) / ratio > 0.08) {
                commensurable = false;
                break;
            }
        }
        if (!commensurable) {
            reasons.push(`${strongCount} 条强谱线，周期之间不成整数比`);
            if (report.hurst > 0.6) {
                reasons.push(`R/S Hurst 指数 ${report.hurst.toFixed(3)}，存在长程持续`);
            }
            return { type: PERIOD_TYPES.QUASI, confidence: "medium", reasons, caveats };
        }
        reasons.push(`${strongCount} 条强谱线近似成整数比，可读作同一基频的谐波族`);
        return { type: PERIOD_TYPES.UNIFORM, confidence: "medium", reasons, caveats };
    }

    if (report.hurst > 0.6) {
        reasons.push(`R/S Hurst 指数 ${report.hurst.toFixed(3)} 大于 0.5（持续），但没有主导尖峰`);
        caveats.push("持续不等于周期：一条趋势也能让 H > 0.5 而并无任何循环");
        return { type: PERIOD_TYPES.QUASI, confidence: "low", reasons, caveats };
    }

    reasons.push(`循环自相关在周期约 ${report.cyclic.peakPeriod.toFixed(1)} 处取得最大`);
    caveats.push("证据不足以在准周期与其它类型之间做出区分");
    return { type: PERIOD_TYPES.QUASI, confidence: "low", reasons, caveats };
};

/** 给定周期在周期图中的功率占比，供对话引用。 */
export const shareAtPeriod: (
    frequencies: ArrayLike<number>,
    power: ArrayLike<number>,
    period: number
) => number = (
    frequencies: ArrayLike<number>,
    power: ArrayLike<number>,
    period: number
): number => bandPowerShare(frequencies, power, period, 0.1);

/** 把报告排成可读多行文本（供对话层引用）。 */
export const describeReport: (report: PeriodicityReport, unit?: string) => string = (
    report: PeriodicityReport,
    unit: string = "采样"
): string => {
    const lines: string[] = [];
    lines.push(`样本数 ${report.length}，均值 ${report.mean.toFixed(3)}，标准差 ${report.standardDeviation.toFixed(3)}`);
    const ac: AutocorrelationPeak | undefined = report.autocorrelationPeaks[0];
    if (ac) {
        lines.push(`平移周期：自相关在滞后 ${ac.lag} (${unit}) 处最强，r = ${ac.value.toFixed(3)}`);
    }
    if (report.peaks.length > 0) {
        const top = report.peaks[0];
        lines.push(`旋转周期：主峰在 ${top.period.toFixed(2)} (${unit})，占功率 ${(report.topPeakShare * 100).toFixed(1)}%`);
        if (report.peaks.length > 1) {
            const others: string = report.peaks
                .slice(1, 4)
                .map((p) => `${p.period.toFixed(1)}`)
                .join("、");
            lines.push(`次峰周期：${others}`);
        }
    }
    lines.push(`缩放周期：R/S Hurst = ${report.hurst.toFixed(4)}（${report.hurst > 0.5 ? "持续" : "反持续"}）`);
    lines.push(`循环频率：主循环周期约 ${report.cyclic.peakPeriod.toFixed(2)} (${unit})`);
    lines.push(`判定：${report.diagnosis.type}（置信度 ${report.diagnosis.confidence}）`);
    return lines.join("\n");
};
