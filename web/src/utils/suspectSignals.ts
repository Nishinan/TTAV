// Data-quality diagnostics ("suspect samples") — a second diagnostic axis
// alongside the distortion lens. Where the lens asks "is the projection
// lying to me here?", these signals ask "is the *data/label* suspicious
// here?". Deliberately kept as separate explainable numbers per signal
// (confidence/variability/correctness, purity, flip count, ...) rather than
// one opaque composite score — the user picks a lens and sees why a sample
// was flagged, not just a rank.
//
// Classification signals need per-epoch prediction/predProbability, which
// loadSingleEpoch only populates when taskType === 'Classification'.
// Code-Retrieval signals need `alignment` ground-truth clusters, which the
// caller must have fetched via BackendAPI.getAlignment first.
import { softmax } from '../component/utils';

type ClassificationEpochData = {
    prediction: number[];
    predProbability: number[][];
};

type NeighborEpochData = {
    originalNeighbors: number[][];
    indexList: number[];
};

type ProjectionEpochData = {
    projection: number[][];
    indexList: number[];
};

function positionOf(indexList: number[] | undefined, rawId: number): number {
    if (!Array.isArray(indexList) || indexList.length === 0) return rawId;
    const pos = indexList.indexOf(rawId);
    return pos >= 0 ? pos : rawId;
}

// ---------------------------------------------------------------------------
// Classification signals
// ---------------------------------------------------------------------------

export type ConfidentlyWrongRow = {
    id: number;
    confidence: number;      // softmax prob assigned to the (wrong) prediction
    predictedLabel: number;
    trueLabel: number;
};

// The strongest single-point signal for a mislabeled sample: the model is
// both wrong AND sure of it.
export function computeConfidentlyWrong(
    epochData: ClassificationEpochData | undefined,
    inherentLabelData: number[],
    confidenceThreshold: number = 0.9,
): ConfidentlyWrongRow[] {
    if (!epochData?.predProbability) return [];
    const rows: ConfidentlyWrongRow[] = [];
    epochData.predProbability.forEach((rawProbs, id) => {
        const trueLabel = inherentLabelData[id];
        if (trueLabel === undefined || !Array.isArray(rawProbs)) return;
        const predictedLabel = epochData.prediction[id];
        if (predictedLabel === undefined || predictedLabel === trueLabel) return;
        const probs = softmax(rawProbs);
        const confidence = probs[predictedLabel] ?? 0;
        if (confidence >= confidenceThreshold) {
            rows.push({ id, confidence, predictedLabel, trueLabel });
        }
    });
    return rows.sort((a, b) => b.confidence - a.confidence);
}

export type OscillationRow = {
    id: number;
    flips: number;
    epochsSeen: number;
};

// Samples whose predicted label keeps flipping across epochs — decision
// boundary instability, often correlated with hard or mislabeled examples.
export function computeOscillation(
    allEpochData: Record<number, ClassificationEpochData>,
): OscillationRow[] {
    const epochs = Object.keys(allEpochData).map(Number).sort((a, b) => a - b);
    if (epochs.length < 2) return [];

    const flips = new Map<number, number>();
    const seen = new Map<number, number>();
    for (let e = 1; e < epochs.length; e += 1) {
        const prevPred = allEpochData[epochs[e - 1]]?.prediction;
        const curPred = allEpochData[epochs[e]]?.prediction;
        if (!Array.isArray(prevPred) || !Array.isArray(curPred)) continue;
        const n = Math.min(prevPred.length, curPred.length);
        for (let i = 0; i < n; i += 1) {
            seen.set(i, (seen.get(i) ?? 0) + 1);
            if (prevPred[i] !== curPred[i]) flips.set(i, (flips.get(i) ?? 0) + 1);
        }
    }

    const rows: OscillationRow[] = [];
    flips.forEach((count, id) => {
        if (count > 0) rows.push({ id, flips: count, epochsSeen: seen.get(id) ?? 0 });
    });
    return rows.sort((a, b) => b.flips - a.flips);
}

export type HdImpurityRow = {
    id: number;
    purity: number;          // fraction of HD top-k neighbors sharing this sample's label
    sameLabelCount: number;
    total: number;
};

// Label-purity in HD space (not 2D!) — low purity means a sample's true
// high-dimensional neighbors mostly disagree with its label: a strong,
// projection-independent hint of a mislabeled or genuinely ambiguous sample.
export function computeHdImpurity(
    epochData: NeighborEpochData | undefined,
    inherentLabelData: number[],
    k: number,
): HdImpurityRow[] {
    if (!epochData?.originalNeighbors) return [];
    const indexList = epochData.indexList ?? [];
    const rows: HdImpurityRow[] = [];

    epochData.originalNeighbors.forEach((hdPositions, pos) => {
        if (!Array.isArray(hdPositions) || hdPositions.length === 0) return;
        const rawId = indexList.length > 0 ? indexList[pos] : pos;
        const ownLabel = inherentLabelData[rawId];
        if (ownLabel === undefined) return;

        const slice = hdPositions.slice(0, k);
        let same = 0;
        slice.forEach((nPos) => {
            const nRaw = indexList.length > 0 ? indexList[nPos] : nPos;
            if (inherentLabelData[nRaw] === ownLabel) same += 1;
        });
        rows.push({ id: rawId, purity: same / slice.length, sameLabelCount: same, total: slice.length });
    });

    return rows.sort((a, b) => a.purity - b.purity);
}

export type CartographyRow = {
    id: number;
    confidence: number;   // mean softmax prob of the TRUE label across loaded epochs
    variability: number;  // std of that confidence across epochs
    correctness: number;  // fraction of loaded epochs predicted correctly
};

// Dataset Cartography (Swayamdipta et al. 2020): every sample gets three
// training-dynamics numbers instead of a single verdict. The two MVP lenses
// built on top (hard-to-learn / ambiguous) are just different sort orders
// over the same three numbers — nothing hidden in a composite score.
export function computeCartography(
    allEpochData: Record<number, ClassificationEpochData>,
    inherentLabelData: number[],
): CartographyRow[] {
    const epochs = Object.keys(allEpochData).map(Number).sort((a, b) => a - b);
    const n = inherentLabelData.length;
    const confSum = new Float64Array(n);
    const confSq = new Float64Array(n);
    const correctCount = new Int32Array(n);
    const seen = new Int32Array(n);

    for (const e of epochs) {
        const ed = allEpochData[e];
        if (!ed?.predProbability) continue;
        for (let i = 0; i < n; i += 1) {
            const trueLabel = inherentLabelData[i];
            const rawProbs = ed.predProbability[i];
            if (trueLabel === undefined || !Array.isArray(rawProbs)) continue;
            const probs = softmax(rawProbs);
            const c = probs[trueLabel] ?? 0;
            confSum[i] += c;
            confSq[i] += c * c;
            seen[i] += 1;
            if (ed.prediction[i] === trueLabel) correctCount[i] += 1;
        }
    }

    const rows: CartographyRow[] = [];
    for (let i = 0; i < n; i += 1) {
        if (seen[i] === 0) continue;
        const mean = confSum[i] / seen[i];
        const variance = Math.max(0, confSq[i] / seen[i] - mean * mean);
        rows.push({ id: i, confidence: mean, variability: Math.sqrt(variance), correctness: correctCount[i] / seen[i] });
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Code-Retrieval (alignment) signals
// ---------------------------------------------------------------------------
// `clusters` are ground-truth alignment groups from BackendAPI.getAlignment
// (union-find over dataset/align.json's ground_truth_pairs). Most clusters
// are size-2 pairs; the code below stays generic over larger groups but only
// ever compares the first two members as the representative pair, since
// that is what align.json actually encodes today.

export type PartnerRankRow = {
    pairIds: [number, number];
    rank: number | null;   // 1-based rank within HD top-k, or null if beyond it
    withinTopK: boolean;
};

// Approximate "is my ground-truth partner even in my HD neighborhood" check.
// Precision is capped at k: a partner beyond top-k is reported as "beyond
// top-k", not a precise rank — getting an exact rank needs HD distances,
// which the frontend does not have (backend-only follow-up).
export function computePartnerRankIssues(
    clusters: number[][],
    epochData: NeighborEpochData | undefined,
    k: number,
): PartnerRankRow[] {
    if (!epochData?.originalNeighbors || !Array.isArray(clusters)) return [];
    const indexList = epochData.indexList ?? [];

    const rows: PartnerRankRow[] = [];
    for (const cluster of clusters) {
        if (!Array.isArray(cluster) || cluster.length < 2) continue;
        const [a, b] = cluster;
        const posA = positionOf(indexList, a);
        const hdList = epochData.originalNeighbors[posA];
        if (!Array.isArray(hdList)) continue;

        const posB = positionOf(indexList, b);
        const sliceTopK = hdList.slice(0, k);
        const rankIdx = sliceTopK.indexOf(posB);
        rows.push({
            pairIds: [a, b],
            rank: rankIdx >= 0 ? rankIdx + 1 : null,
            withinTopK: rankIdx >= 0,
        });
    }

    // Worst first: beyond-top-k pairs, then by rank descending (closest to falling out first).
    return rows.sort((x, y) => {
        if (x.withinTopK !== y.withinTopK) return x.withinTopK ? 1 : -1;
        return (y.rank ?? 0) - (x.rank ?? 0);
    });
}

export type PairDistanceTrendRow = {
    pairIds: [number, number];
    firstDist: number;
    lastDist: number;
    delta: number;   // lastDist - firstDist; positive = drifting apart
};

// A ground-truth pair should converge in LD space as training progresses.
// Pairs still drifting apart across the loaded epoch range are suspects —
// either a bad pair in the alignment file or a training/negative-sampling
// problem, not a projection artifact (LD distance is read directly, no
// re-projection involved).
export function computePairDistanceTrend(
    clusters: number[][],
    allEpochData: Record<number, ProjectionEpochData>,
): PairDistanceTrendRow[] {
    const epochs = Object.keys(allEpochData).map(Number).sort((a, b) => a - b);
    if (epochs.length < 2 || !Array.isArray(clusters)) return [];
    const first = allEpochData[epochs[0]];
    const last = allEpochData[epochs[epochs.length - 1]];

    const distanceBetween = (ed: ProjectionEpochData, a: number, b: number): number | null => {
        const pa = ed.projection[positionOf(ed.indexList, a)];
        const pb = ed.projection[positionOf(ed.indexList, b)];
        if (!Array.isArray(pa) || !Array.isArray(pb)) return null;
        return Math.hypot(pa[0] - pb[0], pa[1] - pb[1]);
    };

    const rows: PairDistanceTrendRow[] = [];
    for (const cluster of clusters) {
        if (!Array.isArray(cluster) || cluster.length < 2) continue;
        const [a, b] = cluster;
        const d0 = distanceBetween(first, a, b);
        const d1 = distanceBetween(last, a, b);
        if (d0 == null || d1 == null) continue;
        rows.push({ pairIds: [a, b], firstDist: d0, lastDist: d1, delta: d1 - d0 });
    }

    return rows.sort((x, y) => y.delta - x.delta); // most-diverging first
}

export type NeighborhoodImpurityRow = {
    pairIds: [number, number];
    purity: number;   // fraction of member a's HD top-k that share a's alignment cluster
};

// Generalization of partner-rank to clusters bigger than a pair: how much of
// a member's HD neighborhood actually belongs to its own alignment group.
export function computeAlignmentNeighborhoodImpurity(
    clusters: number[][],
    epochData: NeighborEpochData | undefined,
    k: number,
): NeighborhoodImpurityRow[] {
    if (!epochData?.originalNeighbors || !Array.isArray(clusters)) return [];
    const indexList = epochData.indexList ?? [];

    const rows: NeighborhoodImpurityRow[] = [];
    for (const cluster of clusters) {
        if (!Array.isArray(cluster) || cluster.length < 2) continue;
        const [a, b] = cluster;
        const clusterSet = new Set(cluster);
        const posA = positionOf(indexList, a);
        const hdList = epochData.originalNeighbors[posA];
        if (!Array.isArray(hdList) || hdList.length === 0) continue;

        const slice = hdList.slice(0, k);
        let inCluster = 0;
        slice.forEach((nPos) => {
            const nRaw = indexList.length > 0 ? indexList[nPos] : nPos;
            if (clusterSet.has(nRaw)) inCluster += 1;
        });
        rows.push({ pairIds: [a, b], purity: inCluster / slice.length });
    }

    return rows.sort((x, y) => x.purity - y.purity); // worst purity first
}
