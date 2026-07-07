import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { EmbeddingView, type EmbeddingViewProps, type DataPoint, type ViewportState } from 'embedding-atlas/react';
import { useDefaultStore } from "../state/state.unified";
import { transferArray2Color } from './utils';
import { computeProjectionNeighborPositionsForPoint, convertNeighborPositionsToRawIndices, rawIndexToProjectionPosition } from '../utils/neighborDiagnostics';

// ---------------------------------------------------------------------------
// RefineStatusBadge — glass pill overlay on top-right of the canvas.
// Reads refineStatus and refineProgress from Zustand.
// ---------------------------------------------------------------------------
function RefineStatusBadge() {
    const { refineStatus, refineProgress } = useDefaultStore(['refineStatus', 'refineProgress']);
    if (refineStatus === 'idle') return null;

    const isDone = refineStatus === 'done';
    const { completed, total } = refineProgress;
    const stepsLabel = total > 0 ? ` ${completed} / ${total}` : completed > 0 ? ` ${completed}` : '';

    return (
        <div style={{
            position: 'absolute',
            top: 10,
            right: 12,
            zIndex: 200,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 10px',
            borderRadius: 20,
            background: 'rgba(255,255,255,0.18)',
            backdropFilter: 'blur(8px)',
            WebkitBackdropFilter: 'blur(8px)',
            border: `1px solid ${isDone ? 'var(--color-success, #22c55e)' : 'var(--accent-blue, #3278F0)'}`,
            boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
            fontSize: 11,
            fontWeight: 600,
            color: isDone ? 'var(--color-success, #22c55e)' : 'var(--accent-blue, #3278F0)',
            pointerEvents: 'none',
            userSelect: 'none',
            whiteSpace: 'nowrap',
        }}>
            {isDone ? (
                <>
                    <span style={{ fontSize: 12 }}>✓</span>
                    <span>Refined</span>
                </>
            ) : (
                <>
                    <span style={{
                        display: 'inline-block',
                        width: 7,
                        height: 7,
                        borderRadius: '50%',
                        background: 'var(--accent-blue, #3278F0)',
                        animation: 'ttav-pulse 1.2s ease-in-out infinite',
                    }} />
                    <span>Refining{stepsLabel}</span>
                </>
            )}
        </div>
    );
}

// ---------------------------------------------------------------------------
// NeighborOverlay — module-scope so the class reference is stable across
// renders. embedding-atlas calls update() rather than creating new instances.
// Box selection is handled by a React overlay div in ChartComponent instead.
// ---------------------------------------------------------------------------

// Compute the SVG path string for the union outline of axis-aligned rectangles.
// Uses coordinate compression + directed boundary edges (clockwise in screen y-down).
function unionRectsPath(rects: Array<{ x1: number; y1: number; x2: number; y2: number }>): string {
    if (rects.length === 0) return '';
    const rs = rects.map(r => ({
        x1: Math.min(r.x1, r.x2), y1: Math.min(r.y1, r.y2),
        x2: Math.max(r.x1, r.x2), y2: Math.max(r.y1, r.y2),
    }));
    const xs = [...new Set(rs.flatMap(r => [r.x1, r.x2]))].sort((a, b) => a - b);
    const ys = [...new Set(rs.flatMap(r => [r.y1, r.y2]))].sort((a, b) => a - b);
    const W = xs.length - 1, H = ys.length - 1;
    const cell = (xi: number, yi: number): boolean =>
        xi >= 0 && xi < W && yi >= 0 && yi < H &&
        rs.some(r => {
            const cx = (xs[xi] + xs[xi + 1]) / 2, cy = (ys[yi] + ys[yi + 1]) / 2;
            return cx > r.x1 && cx < r.x2 && cy > r.y1 && cy < r.y2;
        });

    // Directed boundary half-edges for CW exterior tracing (screen y-down):
    //   top edge of filled cell → L→R, bottom edge → R→L
    //   right edge of filled cell → T→B, left edge → B→T
    type E = [number, number, number, number]; // x1,y1 → x2,y2
    const edges: E[] = [];
    for (let yi = 0; yi <= H; yi++) {
        for (let xi = 0; xi < W; xi++) {
            const a = cell(xi, yi - 1), b = cell(xi, yi);
            if (!a && b) edges.push([xs[xi], ys[yi], xs[xi + 1], ys[yi]]);       // top  L→R
            else if (a && !b) edges.push([xs[xi + 1], ys[yi], xs[xi], ys[yi]]); // bot  R→L
        }
    }
    for (let xi = 0; xi <= W; xi++) {
        for (let yi = 0; yi < H; yi++) {
            const a = cell(xi - 1, yi), b = cell(xi, yi);
            if (a && !b) edges.push([xs[xi], ys[yi], xs[xi], ys[yi + 1]]);       // right T→B
            else if (!a && b) edges.push([xs[xi], ys[yi + 1], xs[xi], ys[yi]]); // left  B→T
        }
    }

    // Chain into closed polygon paths via start-point lookup
    const pk = (x: number, y: number) => `${x},${y}`;
    const nextMap = new Map<string, E>();
    for (const e of edges) nextMap.set(pk(e[0], e[1]), e);

    const used = new Set<string>();
    const paths: string[] = [];
    for (const start of edges) {
        if (used.has(pk(start[0], start[1]))) continue;
        const pts: string[] = [];
        let e: E | undefined = start;
        while (e && !used.has(pk(e[0], e[1]))) {
            used.add(pk(e[0], e[1]));
            pts.push(`${e[0].toFixed(1)},${e[1].toFixed(1)}`);
            e = nextMap.get(pk(e[2], e[3]));
        }
        if (pts.length >= 3) paths.push(`M ${pts.join(' L ')} Z`);
    }
    return paths.join(' ');
}
class NeighborOverlay {
    private el: HTMLDivElement | null = null;
    private svg: SVGSVGElement | null = null;
    private props: any;
    private proxy: any;
    private handleClickBound: (e: MouseEvent) => void;
    private defs: SVGDefsElement | null = null;

    constructor(target: HTMLDivElement, props: any) {
        this.el = target;
        this.props = props;
        this.proxy = props.proxy;
        this.handleClickBound = this.handleClick.bind(this);
        this.mount();
    }

    mount() {
        if (!this.el) return;
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('width', String(this.proxy.width));
        this.svg.setAttribute('height', String(this.proxy.height));
        this.svg.style.display = 'block';
        this.el.appendChild(this.svg);
        this.svg.addEventListener('click', this.handleClickBound);
        this.defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const makeMarker = (id: string, color: string) => {
            const m = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
            m.setAttribute('id', id);
            m.setAttribute('viewBox', '0 0 10 10');
            m.setAttribute('markerUnits', 'userSpaceOnUse');
            m.setAttribute('markerWidth', '9');
            m.setAttribute('markerHeight', '9');
            m.setAttribute('refX', '8');
            m.setAttribute('refY', '5');
            m.setAttribute('orient', 'auto');
            const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            p.setAttribute('d', 'M0,0 L10,5 L0,10 Z');
            p.setAttribute('fill', color);
            m.appendChild(p);
            return m;
        };
        this.defs.appendChild(makeMarker('trail-arrow', '#7F8C8D'));
        this.svg.appendChild(this.defs);
        this.render();
    }

    clear() {
        if (this.svg) {
            const children = Array.from(this.svg.childNodes);
            for (const child of children) {
                if ((child as Element).nodeName.toLowerCase() !== 'defs') {
                    this.svg.removeChild(child);
                }
            }
        }
    }

    // Find the nearest rendered data point to a screen-space mouse event.
    private findNearestId(e: MouseEvent): number | null {
        if (!this.svg) return null;
        const rect = this.svg.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        let minD2 = Infinity;
        let minIdx = -1;
        for (let i = 0; i < this.props.dataX.length; i++) {
            const loc = this.proxy.location(this.props.dataX[i], this.props.dataY[i]);
            const dx = loc.x - sx;
            const dy = loc.y - sy;
            const d2 = dx * dx + dy * dy;
            if (d2 < minD2) { minD2 = d2; minIdx = i; }
        }
        if (minIdx >= 0 && minD2 <= 100) return this.props.idsByPos[minIdx] as number;
        return null;
    }

    handleClick(e: MouseEvent) {
        // When box-select overlay is active the React div sits above this SVG
        // and captures all events — this handler only fires in normal mode.

        // Ctrl/Cmd+Click: deselect nearest point from both primary and secondary.
        if (e.ctrlKey || e.metaKey) {
            const id = this.findNearestId(e);
            if (id !== null) {
                const newSelected = (this.props.selectedIndices as number[] || []).filter(i => i !== id);
                const newSecondary = (this.props.secondaryIndices as number[] || []).filter(i => i !== id);
                this.props.setSelectedIndices?.(newSelected);
                this.props.setSecondaryIndices?.(newSecondary);
            }
            return;
        }

        // Normal click: select nearest point.
        const id = this.findNearestId(e);
        if (id !== null && this.props.setSelectedIndices) {
            this.props.setSelectedIndices([id]);
        }
    }

    render() {
        if (!this.svg) return;
        this.clear();
        const { dataX, dataY, pointSize, revealOriginalNeighbors, revealProjectionNeighbors } = this.props;

        // Support both single-center (legacy) and multi-center (multi-focus) modes.
        const multiGroups: Array<{ center: any; hdOnly: number[]; ldOnly: number[]; overlap: number[] }> =
            this.props.multiCenterGroups ?? [];
        const groups = multiGroups.length > 0
            ? multiGroups
            : (this.props.center
                ? [{ center: this.props.center, hdOnly: this.props.hdOnly ?? [], ldOnly: this.props.ldOnly ?? [], overlap: this.props.overlap ?? [] }]
                : []);

        const STYLE_HD_ONLY = { color: '#E74C3C', lineWidth: 1.6, ringWidth: 2.1, lineOpacity: 0.95, ringOpacity: 0.95 };
        const STYLE_LD_ONLY = { color: '#2E86DE', lineWidth: 1.6, ringWidth: 2.1, lineOpacity: 0.95, ringOpacity: 0.95 };
        const STYLE_OVERLAP = { color: '#95A5A6', lineWidth: 1.2, ringWidth: 1.8, lineOpacity: 0.75, ringOpacity: 0.85 };

        if (groups.length > 0 && (revealOriginalNeighbors || revealProjectionNeighbors)) {
            const neighborGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');

            const drawNeighborFrom = (
                fromLoc: { x: number; y: number },
                nid: number,
                style: { color: string; lineWidth: number; ringWidth: number; lineOpacity?: number; ringOpacity?: number }
            ) => {
                let x: number, y: number;
                const renderedPos = this.props.posMap?.get(nid);
                if (renderedPos != null) {
                    x = dataX[renderedPos];
                    y = dataY[renderedPos];
                } else {
                    const projectionPos = rawIndexToProjectionPosition(nid, this.props.indexList);
                    const coord = this.props.fullProjection?.[projectionPos];
                    if (!coord) return;
                    x = coord[0];
                    y = coord[1];
                }
                const loc = this.proxy.location(x, y);
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', String(fromLoc.x));
                line.setAttribute('y1', String(fromLoc.y));
                line.setAttribute('x2', String(loc.x));
                line.setAttribute('y2', String(loc.y));
                line.setAttribute('stroke', style.color);
                line.setAttribute('stroke-width', String(style.lineWidth));
                line.setAttribute('stroke-linecap', 'round');
                if (style.lineOpacity != null) line.setAttribute('stroke-opacity', String(style.lineOpacity));
                neighborGroup.appendChild(line);
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', String(loc.x));
                circle.setAttribute('cy', String(loc.y));
                circle.setAttribute('r', String(pointSize + 1.5));
                circle.setAttribute('fill', 'none');
                circle.setAttribute('stroke', style.color);
                circle.setAttribute('stroke-width', String(style.ringWidth));
                if (style.ringOpacity != null) circle.setAttribute('stroke-opacity', String(style.ringOpacity));
                neighborGroup.appendChild(circle);
            };

            for (const group of groups) {
                const groupCenterLoc = this.proxy.location(group.center.x, group.center.y);
                if (!groupCenterLoc) continue;
                if (revealOriginalNeighbors || revealProjectionNeighbors)
                    group.overlap.forEach((nid: number) => drawNeighborFrom(groupCenterLoc, nid, STYLE_OVERLAP));
                if (revealOriginalNeighbors)
                    group.hdOnly.forEach((nid: number) => drawNeighborFrom(groupCenterLoc, nid, STYLE_HD_ONLY));
                if (revealProjectionNeighbors)
                    group.ldOnly.forEach((nid: number) => drawNeighborFrom(groupCenterLoc, nid, STYLE_LD_ONLY));
            }
            this.svg.appendChild(neighborGroup);

            // Focus-point center circles for all groups
            for (const group of groups) {
                const groupCenterLoc = this.proxy.location(group.center.x, group.center.y);
                if (!groupCenterLoc) continue;
                const centerCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                centerCircle.setAttribute('cx', String(groupCenterLoc.x));
                centerCircle.setAttribute('cy', String(groupCenterLoc.y));
                centerCircle.setAttribute('r', String(pointSize + 2));
                centerCircle.setAttribute('fill', 'none');
                centerCircle.setAttribute('stroke', '#666');
                centerCircle.setAttribute('stroke-width', '2');
                this.svg.appendChild(centerCircle);
            }
            if (this.props.showTrail) {
                const trailGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                const epochs = this.props.availableEpochs || [];
                const currentIdx = epochs.indexOf(this.props.currentEpoch);
                const centerId = this.props.center?.identifier as number;
                if (typeof centerId === 'number') {
                    const points: { x: number; y: number }[] = [];
                    for (let i = 0; i <= currentIdx; i++) {
                        const ep = epochs[i];
                        const epData = this.props.allEpochData?.[ep];
                        const centerPos = rawIndexToProjectionPosition(centerId, epData?.indexList);
                        const coord = epData?.projection?.[centerPos];
                        if (!coord) continue;
                        const locp = this.proxy.location(coord[0], coord[1]);
                        points.push({ x: locp.x, y: locp.y });
                    }
                    for (let i = 0; i < points.length; i++) {
                        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                        c.setAttribute('cx', String(points[i].x));
                        c.setAttribute('cy', String(points[i].y));
                        c.setAttribute('r', String(Math.max(3, pointSize + 1)));
                        c.setAttribute('fill', '#7F8C8D');
                        c.setAttribute('fill-opacity', '0.85');
                        c.setAttribute('stroke', '#7F8C8D');
                        c.setAttribute('stroke-width', '0.5');
                        trailGroup.appendChild(c);
                    }
                    for (let i = 1; i < points.length; i++) {
                        const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                        l.setAttribute('x1', String(points[i - 1].x));
                        l.setAttribute('y1', String(points[i - 1].y));
                        l.setAttribute('x2', String(points[i].x));
                        l.setAttribute('y2', String(points[i].y));
                        l.setAttribute('stroke', '#7F8C8D');
                        l.setAttribute('stroke-width', '2');
                        l.setAttribute('stroke-dasharray', '6 3');
                        l.setAttribute('stroke-linecap', 'round');
                        l.setAttribute('stroke-opacity', '0.9');
                        l.setAttribute('marker-end', 'url(#trail-arrow)');
                        trailGroup.appendChild(l);
                    }
                }
                this.svg.appendChild(trailGroup);
            }
        }

        // Secondary boxes (tiered mode): draw union polygon outline
        const secondaryBoxes: Array<[number, number, number, number]> = this.props.secondaryBoxes || [];
        if (secondaryBoxes.length > 0) {
            // Convert each stored data-coord box to SVG screen coords via proxy,
            // then compute the union polygon outline.
            const screenRects = secondaryBoxes.map(([dx1, dy1, dx2, dy2]) => {
                // dataY1 > dataY2 (y-axis flipped), so (dx1,dy1) = top-left in screen
                const tl = this.proxy.location(dx1, dy1);
                const br = this.proxy.location(dx2, dy2);
                return { x1: tl.x, y1: tl.y, x2: br.x, y2: br.y };
            });
            const d = unionRectsPath(screenRects);
            if (d) {
                const boxGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('d', d);
                path.setAttribute('fill', 'rgba(124,58,237,0.06)');
                path.setAttribute('stroke', '#7c3aed');
                path.setAttribute('stroke-width', '1.5');
                path.setAttribute('stroke-dasharray', '6 3');
                path.setAttribute('pointer-events', 'none');
                boxGroup.appendChild(path);
                this.svg.appendChild(boxGroup);
            }
        }

        // Primary selected indices: gold / dark rings
        const selectedGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        const selectedSet = new Set<number>(this.props.selectedIndices ?? []);
        selectedSet.forEach((selectedId: number) => {
            const pos = this.props.posMap.get(selectedId);
            if (pos == null) return;
            const loc = this.proxy.location(dataX[pos], dataY[pos]);
            const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            ring.setAttribute('cx', String(loc.x));
            ring.setAttribute('cy', String(loc.y));
            ring.setAttribute('r', String(pointSize + 3));
            ring.setAttribute('fill', 'none');
            ring.setAttribute('stroke', selectedId === this.props.center?.identifier ? '#111827' : '#f59e0b');
            ring.setAttribute('stroke-width', selectedId === this.props.center?.identifier ? '2.5' : '2');
            selectedGroup.appendChild(ring);
        });
        this.svg.appendChild(selectedGroup);

        if (this.props.showLabel || this.props.showIndex || selectedSet.size > 0) {
            const textGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            const occupiedBoxes: { x: number, y: number, width: number, height: number }[] = [];
            const padding = 2;
            const baseCharWidth = 6;
            const baseCharHeight = 10;

            const renderLabel = (_id: number, loc: { x: number; y: number }, content: string, forceVisible: boolean) => {
                const fontSize = forceVisible ? 13 : 10;
                const charWidth = forceVisible ? 7.5 : baseCharWidth;
                const charHeight = forceVisible ? 13 : baseCharHeight;
                const boxWidth = content.length * charWidth;
                const boxHeight = charHeight;
                const baseOffset = pointSize + 2;
                const candidateOffsets = forceVisible
                    ? [
                        { dx: baseOffset, dy: -baseOffset },
                        { dx: baseOffset, dy: charHeight + 4 },
                        { dx: -(boxWidth + baseOffset), dy: -baseOffset },
                        { dx: -(boxWidth + baseOffset), dy: charHeight + 4 },
                        { dx: -(boxWidth / 2), dy: -(pointSize + 10) },
                        { dx: -(boxWidth / 2), dy: charHeight + pointSize + 6 },
                    ]
                    : [{ dx: baseOffset, dy: -baseOffset }];

                let chosen: { labelX: number; labelY: number; boxX: number; boxY: number } | null = null;
                for (const candidate of candidateOffsets) {
                    const labelX = loc.x + candidate.dx;
                    const labelY = loc.y + candidate.dy;
                    const boxX = labelX;
                    const boxY = labelY - charHeight;
                    let collision = false;
                    for (const box of occupiedBoxes) {
                        if (
                            boxX < box.x + box.width + padding &&
                            boxX + boxWidth + padding > box.x &&
                            boxY < box.y + box.height + padding &&
                            boxY + boxHeight + padding > box.y
                        ) { collision = true; break; }
                    }
                    if (!collision) { chosen = { labelX, labelY, boxX, boxY }; break; }
                }
                if (!chosen) {
                    if (!forceVisible) return;
                    const fallbackLabelX = loc.x + baseOffset;
                    const fallbackLabelY = loc.y - baseOffset;
                    chosen = { labelX: fallbackLabelX, labelY: fallbackLabelY, boxX: fallbackLabelX, boxY: fallbackLabelY - charHeight };
                }
                const textEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                textEl.setAttribute('x', String(chosen.labelX));
                textEl.setAttribute('y', String(chosen.labelY));
                textEl.setAttribute('fill', forceVisible ? '#111827' : '#000');
                textEl.setAttribute('font-size', String(fontSize));
                textEl.setAttribute('font-family', 'Console, monospace');
                if (forceVisible) {
                    textEl.setAttribute('font-weight', '700');
                    textEl.setAttribute('paint-order', 'stroke');
                    textEl.setAttribute('stroke', '#ffffff');
                    textEl.setAttribute('stroke-width', '3');
                    textEl.setAttribute('stroke-linejoin', 'round');
                }
                textEl.textContent = content;
                textGroup.appendChild(textEl);
                occupiedBoxes.push({ x: chosen.boxX, y: chosen.boxY, width: boxWidth, height: boxHeight });
            };

            selectedSet.forEach((selectedId: number) => {
                const pos = this.props.posMap.get(selectedId);
                if (pos == null) return;
                const loc = this.proxy.location(this.props.dataX[pos], this.props.dataY[pos]);
                const labelTextData = this.props.textData && this.props.textData[selectedId]
                    ? this.props.textData[selectedId]
                    : (this.props.labelDict?.get(this.props.inherentLabelData[selectedId]) ?? '');
                const content = formatPointLabel(selectedId, labelTextData, true, true);
                if (!content) return;
                renderLabel(selectedId, loc, content, true);
            });

            for (let i = 0; i < this.props.dataX.length; i++) {
                const id = this.props.idsByPos[i];
                if (selectedSet.has(id)) continue;
                const loc = this.proxy.location(this.props.dataX[i], this.props.dataY[i]);
                const labelTextData = this.props.textData && this.props.textData[id]
                    ? this.props.textData[id]
                    : (this.props.labelDict?.get(this.props.inherentLabelData[id]) ?? '');
                const content = formatPointLabel(id, labelTextData, this.props.showLabel, this.props.showIndex);
                if (!content) continue;
                renderLabel(id, loc, content, false);
            }
            this.svg.appendChild(textGroup);
        }

    }

    update(nextProps: Partial<any>) {
        this.props = { ...this.props, ...nextProps };
        // Always sync proxy so viewport-to-screen conversion stays current.
        if (this.props.proxy) this.proxy = this.props.proxy;
        if (this.svg) {
            this.svg.setAttribute('width', String(this.proxy.width));
            this.svg.setAttribute('height', String(this.proxy.height));
        }
        this.render();
    }

    destroy() {
        if (this.svg && this.el) this.el.removeChild(this.svg);
        if (this.svg) this.svg.removeEventListener('click', this.handleClickBound);
        this.defs = null;
        this.svg = null;
        this.el = null;
    }
}
// ---------------------------------------------------------------------------

type EmbeddingData = NonNullable<EmbeddingViewProps['data']>;

type PreparedEmbedding = {
    simpleData: EmbeddingData;
    dataPoints: DataPoint[];
    categoryColors: string[] | null;
};

function formatPointLabel(id: number, rawLabel: string, showLabel: boolean, showIndex: boolean): string {
    const normalized = rawLabel.replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim();
    const tokenMatch = normalized.match(/^([PO]\d+):\s*(.*)$/);

    if (tokenMatch) {
        const prefix = tokenMatch[1];
        const tail = tokenMatch[2].replace(/[␠ ]+/g, ' ').trim() || '·';
        if (showLabel && showIndex) {
            return `${prefix}: ${tail}`;
        }
        if (showLabel) {
            return `${prefix}: ${tail}`;
        }
        if (showIndex) {
            return String(id);
        }
        return '';
    }

    if (showLabel && showIndex) {
        return normalized ? `${id}. ${normalized}` : String(id);
    }
    if (showLabel) {
        return normalized;
    }
    if (showIndex) {
        return String(id);
    }
    return '';
}

export const ChartComponent = memo(() => {
    const atlasRef = useRef<HTMLDivElement | null>(null);
    const [dimensions, setDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 });

    const { epoch, allEpochData, globalBounds } = useDefaultStore(["epoch", "allEpochData", "globalBounds"]);
    const { inherentLabelData, colorDict, labelDict, textData } = useDefaultStore(["inherentLabelData", "colorDict", "labelDict", "textData"]);
    const { shownData, index, isFocusMode, focusIndices } = useDefaultStore(["shownData", "index", "isFocusMode", "focusIndices"]);
    const { highlightData } = useDefaultStore(["highlightData"]);

    const { hoveredIndex, setHoveredIndex } = useDefaultStore(["hoveredIndex", "setHoveredIndex"]);
    const { mode } = useDefaultStore(["mode"]);
    const { pointSize } = useDefaultStore(["pointSize"]);
    const { revealOriginalNeighbors, revealProjectionNeighbors } = useDefaultStore(["revealOriginalNeighbors", "revealProjectionNeighbors"]);
    const { showLabel, showIndex } = useDefaultStore(["showLabel", "showIndex"]);
    const { selectedIndices } = useDefaultStore(["selectedIndices"]);
    const { availableEpochs } = useDefaultStore(["availableEpochs"]);
    const { showTrail } = useDefaultStore(["showTrail"]);
    const { setSelectedIndices } = useDefaultStore(["setSelectedIndices"]);
    const { setCurrentViewportBBox } = useDefaultStore(["setCurrentViewportBBox"]);
    const { boxSelectActive, refineFocusType, secondaryIndices, setSecondaryIndices, secondaryBoxes, setSecondaryBoxes } =
        useDefaultStore(["boxSelectActive", "refineFocusType", "secondaryIndices", "setSecondaryIndices", "secondaryBoxes", "setSecondaryBoxes"]);
    const { neighborDisplayIndices } = useDefaultStore(["neighborDisplayIndices"]);
    const { showPreRefine } = useDefaultStore(["showPreRefine"]);

    // B3: non-destructive before/after toggle — when showPreRefine is on, render
    // the pre-refine baseline (originalProjection) as the projection so every
    // downstream derivation (points, neighbors, overlay) reflects the "before".
    const _rawEpochData = allEpochData[epoch];
    const epochData = (showPreRefine && _rawEpochData?.originalProjection)
        ? { ..._rawEpochData, projection: _rawEpochData.originalProjection }
        : _rawEpochData;
    const activePointId = selectedIndices[0] ?? hoveredIndex;

    // plot view helpers
    let [tooltip, setTooltip] = useState<DataPoint | null>(null);
    // selection can be added later when needed
    let [viewportState, setViewportState] = useState<ViewportState | null>(null);

    // observe container size change
    useEffect(() => {
        const node = atlasRef.current;
        if (!node) {
            return;
        }

        const observer = new ResizeObserver(([entry]) => {
            if (!entry) {
                return;
            }
            const { width, height } = entry.contentRect;
            setDimensions((prev) => (
                prev.width === width && prev.height === height
                    ? prev
                    : { width, height }
            ));
        });

        observer.observe(node);

        return () => {
            observer.disconnect();
        };
    }, []);

    // set viewport based on global bounds
    useEffect(() => {
        if (!globalBounds) return;

        const { minX, maxX, minY, maxY } = globalBounds;

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;

        const rangeX = maxX - minX;
        const rangeY = maxY - minY;

        const safeRangeX = rangeX === 0 ? 1 : rangeX;
        const safeRangeY = rangeY === 0 ? 1 : rangeY;

        const padding = 1.5;

        const scaleX = 2 / (safeRangeX * padding);
        const scaleY = 2 / (safeRangeY * padding);

        const scale = Math.min(scaleX, scaleY);

        setViewportState({
            x: centerX,
            y: centerY,
            scale,
        });

    }, [globalBounds]);

    useEffect(() => {
        if (!viewportState || dimensions.width <= 0 || dimensions.height <= 0) {
            setCurrentViewportBBox(null);
            return;
        }

        const safeScale = viewportState.scale && viewportState.scale > 0
            ? viewportState.scale
            : 1;
        const aspect = dimensions.height > 0 ? dimensions.width / dimensions.height : 1;
        const halfHeight = 1 / safeScale;
        const halfWidth = aspect / safeScale;

        setCurrentViewportBBox({
            xMin: viewportState.x - halfWidth,
            xMax: viewportState.x + halfWidth,
            yMin: viewportState.y - halfHeight,
            yMax: viewportState.y + halfHeight,
        });
    }, [dimensions.height, dimensions.width, setCurrentViewportBBox, viewportState]);

    const diagnosticVisibleIds = useMemo(() => {
        if (!epochData || activePointId === undefined) {
            return [] as number[];
        }

        // Both originalNeighbors and projectionNeighbors are indexed by PROJECTION POSITION
        // and contain PROJECTION POSITIONS (not raw indices). Convert accordingly.
        const activePointPos = rawIndexToProjectionPosition(activePointId, epochData.indexList);
        const hdIds = revealOriginalNeighbors
            ? convertNeighborPositionsToRawIndices(
                epochData.originalNeighbors?.[activePointPos] ?? [],
                epochData.indexList
              )
            : [];
        const ldPositions = computeProjectionNeighborPositionsForPoint(
            activePointId,
            epochData.originalProjection ?? epochData.projection,
            epochData.indexList,
            10
        );
        const ldIds = revealProjectionNeighbors
            ? convertNeighborPositionsToRawIndices(ldPositions, epochData.indexList)
            : [];

        return Array.from(new Set<number>([
            activePointId,
            ...selectedIndices,
            ...hdIds,
            ...ldIds,
        ]));
    }, [
        epochData,
        activePointId,
        selectedIndices,
        revealOriginalNeighbors,
        revealProjectionNeighbors,
    ]);

    // filter dataIndices
    const filteredIndices = useMemo(() => {
        if (!epochData) {
            return [] as number[];
        }
        const totalIndices = Array.isArray(epochData.indexList) && epochData.indexList.length > 0
            ? epochData.indexList
            : epochData.projection.map((_, idx) => idx);
        let current = totalIndices;

        if (shownData.length > 0) {
            const shownIndexSet = new Set<number>();
            shownData.forEach((key) => {
                const group = index[key];
                if (Array.isArray(group)) {
                    group.forEach((value) => shownIndexSet.add(value));
                }
            });
            if (shownIndexSet.size > 0) {
                current = current.filter((value) => shownIndexSet.has(value));
            }
        }

        if (isFocusMode && focusIndices.length > 0) {
            const focusSet = new Set<number>([...focusIndices, ...diagnosticVisibleIds]);
            current = current.filter((value) => focusSet.has(value));
        }

        return current;
    }, [diagnosticVisibleIds, epochData, focusIndices, index, isFocusMode, shownData]);

    // Build a set of highlighted point indices based on highlightData toggles.
    // prediction_error: prediction !== ground-truth label
    // prediction_flip:  prediction changed vs. the previous epoch
    const highlightedSet = useMemo<Set<number>>(() => {
        const s = new Set<number>();
        if (!epochData || highlightData.length === 0) return s;

        const prediction = epochData.prediction;
        if (!prediction) return s;

        if (highlightData.includes('prediction_error')) {
            prediction.forEach((pred, idx) => {
                if (pred !== inherentLabelData[idx]) s.add(idx);
            });
        }

        if (highlightData.includes('prediction_flip')) {
            const epochIndex = allEpochData ? Object.keys(allEpochData).map(Number).sort((a, b) => a - b) : [];
            const epochPos = epochIndex.indexOf(epoch);
            if (epochPos > 0) {
                const prevEpoch = epochIndex[epochPos - 1];
                const prevPrediction = allEpochData[prevEpoch]?.prediction;
                if (prevPrediction) {
                    prediction.forEach((pred, idx) => {
                        if (pred !== prevPrediction[idx]) s.add(idx);
                    });
                }
            }
        }

        return s;
    }, [epochData, highlightData, inherentLabelData, allEpochData, epoch]);

    // convert data for embedding view
    const prepared = useMemo<PreparedEmbedding | null>(() => {
        if (!epochData || filteredIndices.length === 0) {
            return null;
        }

        const x = new Float32Array(filteredIndices.length);
        const y = new Float32Array(filteredIndices.length);
        const category = new Uint8Array(filteredIndices.length);
        const categoryColorList: string[] = [];
        const labelToCategoryIndex = new Map<number, number>();

        // Reserve a fixed slot for the highlight colour (bright red) at index 0
        // so highlighted points always render red regardless of their class.
        const HIGHLIGHT_COLOR = '#ff2222';
        const HIGHLIGHT_CATEGORY_IDX = 0;
        const hasHighlights = highlightedSet.size > 0;
        if (hasHighlights) {
            categoryColorList.push(HIGHLIGHT_COLOR); // slot 0
        }

        let dataPoints : DataPoint[] = []

        filteredIndices.forEach((rawIndex, position) => {
            const projectionPos = rawIndexToProjectionPosition(rawIndex, epochData.indexList);
            const [px, py] = epochData.projection[projectionPos] ?? [0, 0];
            x[position] = px;
            y[position] = py;

            // Highlighted points always use slot 0 (red); normal points use their label colour.
            if (hasHighlights && highlightedSet.has(rawIndex)) {
                category[position] = HIGHLIGHT_CATEGORY_IDX;
            } else {
                const label = inherentLabelData[rawIndex] ?? 0;
                const colorTuple = colorDict.get(label);
                // Offset by 1 when highlights are active to leave slot 0 for red
                let categoryIndex = labelToCategoryIndex.get(label);
                if (categoryIndex === undefined) {
                    categoryIndex = categoryColorList.length; // next available slot
                    labelToCategoryIndex.set(label, categoryIndex);
                    const colorString = transferArray2Color(colorTuple, 1);
                    categoryColorList.push(colorString);
                }
                category[position] = categoryIndex;
            }

            const label = inherentLabelData[rawIndex] ?? 0;
            dataPoints.push({
                x: px,
                y: py,
                category: label,
                text: `Index: ${rawIndex}\nLabel: ${label}`,
                identifier: rawIndex,
                fields: {}
            })
        });

        const simpleData = {
            x,
            y,
            category: categoryColorList.length > 0 ? category : undefined,
        } as unknown as EmbeddingData;

        return {
            simpleData,
            dataPoints,
            categoryColors: categoryColorList.length > 0 ? categoryColorList : null,
        };
    }, [colorDict, epochData, filteredIndices, inherentLabelData, highlightedSet]);

    const posMap = useMemo(() => {
        const m = new Map<number, number>();
        if (prepared) {
            prepared.dataPoints.forEach((p, i) => {
                m.set(p.identifier as number, i);
            });
        }
        return m;
    }, [prepared]);

    const controlledSelection = useMemo(() => {
        if (!prepared || selectedIndices.length === 0) return null;
        return selectedIndices
            .map((selectedId) => {
                const pos = posMap.get(selectedId);
                return pos == null ? null : prepared.dataPoints[pos];
            })
            .filter((point): point is DataPoint => point !== null);
    }, [posMap, prepared, selectedIndices]);

    useEffect(() => {
        if (!prepared || hoveredIndex === undefined) {
            return;
        }
        const pos = posMap.get(hoveredIndex);
        if (pos === undefined) {
            return;
        }
        setTooltip(prepared.dataPoints[pos] ?? null);
    }, [hoveredIndex, prepared, posMap]);

    const [trailRefresh, setTrailRefresh] = useState(0);
    useEffect(() => { setTrailRefresh((v) => v + 1); }, [selectedIndices]);


    const neighborOverlayProps = useMemo(() => {
        const baseProps = {
            dataX: prepared?.simpleData?.x as Float32Array ?? new Float32Array(0),
            dataY: prepared?.simpleData?.y as Float32Array ?? new Float32Array(0),
            fullProjection: epochData?.projection,
            indexList: epochData?.indexList,
            pointSize,
            revealOriginalNeighbors,
            revealProjectionNeighbors,
            idsByPos: prepared?.dataPoints?.map((p) => p.identifier as number) ?? [],
            showLabel, showIndex, labelDict, textData, inherentLabelData, viewportState,
            showTrail, availableEpochs, allEpochData, currentEpoch: epoch,
            setSelectedIndices, selectedIndices, secondaryIndices, setSecondaryIndices, secondaryBoxes,
        };
        if (!prepared || !epochData) return { ...baseProps, center: null, multiCenterGroups: [] } as any;

        // Determine which point IDs to show neighbor lines for.
        // When points are selected: use neighborDisplayIndices (user-checked subset).
        // When only hovering: use the hovered point.
        const focusIds: number[] = selectedIndices.length > 0
            ? neighborDisplayIndices.filter(i => selectedIndices.includes(i))
            : (activePointId !== undefined ? [activePointId] : []);

        // Use the currently-displayed projection for LD neighbor computation.
        // originalProjection is the pre-refine baseline — using it after refine
        // causes blue lines to be drawn to stale positions from the old layout.
        const ldProjection = epochData.projection;

        // Build neighbor groups for every focus point.
        const multiCenterGroups: Array<{ center: any; hdOnly: number[]; ldOnly: number[]; overlap: number[] }> = [];
        for (const fid of focusIds) {
            const focusPos = rawIndexToProjectionPosition(fid, epochData.indexList);

            const hdPositions: number[] = epochData.originalNeighbors?.[focusPos] ?? [];
            const hdAll = convertNeighborPositionsToRawIndices(hdPositions, epochData.indexList);

            const ldPositions = computeProjectionNeighborPositionsForPoint(fid, ldProjection, epochData.indexList, 10);
            const ldAll = convertNeighborPositionsToRawIndices(ldPositions, epochData.indexList);

            const hdSet = new Set<number>(hdAll);
            const ldSet = new Set<number>(ldAll);

            const centerPos = posMap.get(fid);
            const center = centerPos == null ? null : prepared.dataPoints[centerPos];
            if (!center) continue;

            multiCenterGroups.push({
                center,
                hdOnly: hdAll.filter((nid: number) => !ldSet.has(nid)),
                ldOnly: ldAll.filter((nid: number) => !hdSet.has(nid)),
                overlap: hdAll.filter((nid: number) => ldSet.has(nid)),
            });
        }

        // Primary center (first group) kept for trail rendering.
        const primaryGroup = multiCenterGroups[0] ?? null;

        return {
            ...baseProps,
            center: primaryGroup?.center ?? null,
            hdOnly:  primaryGroup?.hdOnly  ?? [],
            ldOnly:  primaryGroup?.ldOnly  ?? [],
            overlap: primaryGroup?.overlap ?? [],
            multiCenterGroups,
        };
    }, [prepared, epochData, activePointId, selectedIndices, neighborDisplayIndices, posMap, pointSize, revealOriginalNeighbors, revealProjectionNeighbors, showLabel, showIndex, labelDict, textData, inherentLabelData, viewportState, showTrail, availableEpochs, allEpochData, epoch, trailRefresh, secondaryIndices, setSecondaryIndices, secondaryBoxes]);

    // ---- box select overlay state & handlers ----
    const [boxDrag, setBoxDrag] = useState<{ startX: number; startY: number; curX: number; curY: number } | null>(null);

    const handleBoxMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setBoxDrag({ startX: x, startY: y, curX: x, curY: y });
    };

    const handleBoxMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!boxDrag) return;
        const rect = e.currentTarget.getBoundingClientRect();
        setBoxDrag((prev) => prev ? { ...prev, curX: e.clientX - rect.left, curY: e.clientY - rect.top } : null);
    };

    const handleBoxMouseUp = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!boxDrag) return;
        const overlayRect = e.currentTarget.getBoundingClientRect();
        const curX = e.clientX - overlayRect.left;
        const curY = e.clientY - overlayRect.top;
        const x1 = Math.min(boxDrag.startX, curX), x2 = Math.max(boxDrag.startX, curX);
        const y1 = Math.min(boxDrag.startY, curY), y2 = Math.max(boxDrag.startY, curY);
        if (prepared && viewportState) {
            // Use the overlay div's CSS dimensions directly.
            // viewportState.scale is calibrated in CSS pixels
            // (1 data unit = scale * cssHalfH CSS pixels), so we must NOT use
            // the SVG element's physical-pixel width/height attribute here.
            const W = overlayRect.width;
            const H = overlayRect.height;
            if (H <= 0) { setBoxDrag(null); return; }

            const { x: vx, y: vy, scale } = viewportState;
            const halfH = H / 2;
            // CSS → data coordinate conversion
            const toDataX = (cx: number) => vx + (cx - W / 2) / (scale * halfH);
            const toDataY = (cy: number) => vy - (cy - halfH) / (scale * halfH);

            const boxIds: number[] = [];
            for (let i = 0; i < prepared.simpleData.x.length; i++) {
                // Project data point into CSS pixel space
                const px = W / 2 + (prepared.simpleData.x[i] - vx) * scale * halfH;
                const py = halfH - (prepared.simpleData.y[i] - vy) * scale * halfH;
                if (px >= x1 && px <= x2 && py >= y1 && py <= y2)
                    boxIds.push(prepared.dataPoints[i].identifier as number);
            }
            const dataBox: [number, number, number, number] = [
                toDataX(x1), toDataY(y1), toDataX(x2), toDataY(y2),
            ];
            if (refineFocusType === 'tiered') {
                const primarySet = new Set(selectedIndices);
                const existingSecondary = new Set(secondaryIndices);
                const newSecondary = [...secondaryIndices];
                for (const id of boxIds) {
                    if (!primarySet.has(id) && !existingSecondary.has(id)) newSecondary.push(id);
                }
                setSecondaryIndices(newSecondary);
                setSecondaryBoxes([...secondaryBoxes, dataBox]);
            } else {
                setSelectedIndices(boxIds);
                setSecondaryIndices([]);
                setSecondaryBoxes([]);
            }
        }
        setBoxDrag(null);
    };

    // find selected datapoint
    async function querySelection(x: number, y: number, unitDistance: number): Promise<DataPoint | null> {
        if (!prepared) {
            return null;
        }
        let simpleData = prepared.simpleData;
        let minDistance2: number | null = null;
        let minIndex: number | null = null;
        for (let i = 0; i < simpleData.x.length; i++) {
        let d2 = (simpleData.x[i] - x) * (simpleData.x[i] - x) + (simpleData.y[i] - y) * (simpleData.y[i] - y);
        if (minDistance2 == null || d2 < minDistance2) {
            minDistance2 = d2;
            minIndex = i;
        }
        }
        if (minIndex == null || minDistance2 == null || Math.sqrt(minDistance2) > unitDistance * 10) {
            return null;
        }
        return prepared.dataPoints[minIndex];
    }


    const content = prepared ? (
        <EmbeddingView
            data={prepared.simpleData}
            categoryColors={prepared.categoryColors}
            width={dimensions.width || undefined}
            height={dimensions.height || undefined}
            config={{ mode: mode, colorScheme: 'light', pointSize: pointSize }}
            tooltip={tooltip}
            selection={controlledSelection}
            onTooltip={(v) => {
                setHoveredIndex(v ? v.identifier as number : undefined);
                setTooltip(v);
            }}
            viewportState={viewportState}
            onViewportState={(v) => setViewportState(v)}
            querySelection={ querySelection }
            customOverlay={{
                class: NeighborOverlay as any,
                props: { ...neighborOverlayProps, posMap }
            }}
            // [确定性逻辑 3]：确保 EmbeddingView 的选中事件同步到全局 Store
            onSelection={(points) => {
                const ids = points && points.length > 0
                    ? points.map((p) => p.identifier as number)
                    : [];
                console.log("[TTAV] Selection Sync to Store:", ids);
                if (ids.length > 0) {
                    setSelectedIndices(ids);
                }
            }}
        />
    ) : null;

    return (
        <div
            ref={atlasRef}
            style={{
                display: 'flex',
                flexDirection: 'column',
                width: '100%',
                height: '100%',
            }}
        >
            <div style={{ position: 'relative', flex: 1 }}>
                <style>{`@keyframes ttav-pulse{0%,100%{opacity:1}50%{opacity:0.35}}`}</style>
                {content ?? <div style={{ width: '100%', height: '100%' }} />}
                <RefineStatusBadge />
                {boxSelectActive && (
                    <div
                        style={{ position: 'absolute', inset: 0, cursor: 'crosshair', zIndex: 100, userSelect: 'none' }}
                        onMouseDown={handleBoxMouseDown}
                        onMouseMove={handleBoxMouseMove}
                        onMouseUp={handleBoxMouseUp}
                    >
                        {boxDrag && (
                            <div style={{
                                position: 'absolute',
                                left: Math.min(boxDrag.startX, boxDrag.curX),
                                top: Math.min(boxDrag.startY, boxDrag.curY),
                                width: Math.abs(boxDrag.curX - boxDrag.startX),
                                height: Math.abs(boxDrag.curY - boxDrag.startY),
                                border: '1.5px dashed #7c3aed',
                                background: 'rgba(124,58,237,0.08)',
                                pointerEvents: 'none',
                            }} />
                        )}
                    </div>
                )}
            </div>
        </div>
    );
});

export default ChartComponent;
