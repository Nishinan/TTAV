export type NeighborDiagnostics = {
    hdAll: number[];
    ldAll: number[];
    hdOnly: number[];
    ldOnly: number[];
    overlap: number[];
};

function toUniqueValidIntegers(values: number[] | undefined | null): number[] {
    if (!Array.isArray(values)) return [];
    const seen = new Set<number>();
    const result: number[] = [];
    values.forEach((value) => {
        if (!Number.isInteger(value)) return;
        if (seen.has(value)) return;
        seen.add(value);
        result.push(value);
    });
    return result;
}

export function normalizeNeighborRawIndices(
    neighborIds: number[] | undefined | null,
): number[] {
    return toUniqueValidIntegers(neighborIds);
}

export function rawIndexToProjectionPosition(
    rawIndex: number,
    indexList: number[] | undefined | null,
): number {
    const indices = Array.isArray(indexList) ? indexList : [];
    if (indices.length === 0) return rawIndex;
    const pos = indices.indexOf(rawIndex);
    return pos >= 0 ? pos : rawIndex;
}

export function convertNeighborPositionsToRawIndices(
    neighborPositions: number[] | undefined | null,
    indexList: number[] | undefined | null,
): number[] {
    const indices = Array.isArray(indexList) ? indexList : [];
    return toUniqueValidIntegers(neighborPositions).map((arrayPos) => indices[arrayPos] ?? arrayPos);
}

export function convertProjectionNeighborsToRawIndices(
    projectionNeighbors: number[] | undefined | null,
    indexList: number[] | undefined | null,
): number[] {
    return convertNeighborPositionsToRawIndices(projectionNeighbors, indexList);
}

export function computeProjectionNeighborPositionsForPoint(
    targetIndex: number,
    projection: number[][] | undefined | null,
    indexList: number[] | undefined | null,
    k: number = 10,
): number[] {
    if (!Array.isArray(projection) || projection.length === 0 || !Number.isInteger(targetIndex)) {
        return [];
    }

    const ids = Array.isArray(indexList) ? indexList : [];
    let targetPos = ids.length > 0 ? ids.indexOf(targetIndex) : -1;
    if (targetPos < 0 && targetIndex >= 0 && targetIndex < projection.length) {
        // Fallback for views that still pass projection-array positions instead of raw ids.
        targetPos = targetIndex;
    }
    if (targetPos < 0 || targetPos >= projection.length) {
        return [];
    }

    const center = projection[targetPos];
    if (!Array.isArray(center) || center.length < 2) {
        return [];
    }

    const best: Array<{ pos: number; dist2: number }> = [];
    const limit = Math.max(1, Math.min(k, projection.length - 1));

    for (let pos = 0; pos < projection.length; pos += 1) {
        if (pos === targetPos) continue;
        const point = projection[pos];
        if (!Array.isArray(point) || point.length < 2) continue;
        const dx = point[0] - center[0];
        const dy = point[1] - center[1];
        const dist2 = dx * dx + dy * dy;

        if (best.length < limit) {
            best.push({ pos, dist2 });
            best.sort((a, b) => a.dist2 - b.dist2);
            continue;
        }

        if (dist2 >= best[best.length - 1].dist2) continue;
        best[best.length - 1] = { pos, dist2 };
        best.sort((a, b) => a.dist2 - b.dist2);
    }

    return best.map((item) => item.pos);
}

export function classifyNeighborDiagnostics(
    originalNeighbors: number[] | undefined | null,
    projectionNeighbors: number[] | undefined | null,
    indexList: number[] | undefined | null,
): NeighborDiagnostics {
    const hdAll = convertNeighborPositionsToRawIndices(originalNeighbors, indexList);
    const ldAll = convertNeighborPositionsToRawIndices(projectionNeighbors, indexList);

    const hdSet = new Set(hdAll);
    const ldSet = new Set(ldAll);

    return {
        hdAll,
        ldAll,
        hdOnly: hdAll.filter((id) => !ldSet.has(id)),
        ldOnly: ldAll.filter((id) => !hdSet.has(id)),
        overlap: hdAll.filter((id) => ldSet.has(id)),
    };
}
