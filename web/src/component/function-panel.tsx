import { AutoComplete, Input, InputNumber, List, Tag, RefSelectProps, Checkbox, Switch, Select, Slider, Button, Tooltip, message } from 'antd';
import { useDefaultStore, FocusMode, RefineFocusType } from '../state/state.unified';
import * as BackendAPI from '../communication/backend';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ComponentBlock, FunctionalBlock } from './custom/basic-components';
import { styled } from 'styled-components';
import { SyncOutlined } from '@ant-design/icons';
import { BoxSelect, MousePointer2, X, XCircle, RefreshCw } from 'lucide-react';
import { computeProjectionNeighborPositionsForPoint, convertNeighborPositionsToRawIndices, rawIndexToProjectionPosition, computeAllPointsNeighborPreservation } from '../utils/neighborDiagnostics';
import {
    computeConfidentlyWrong, computeOscillation, computeHdImpurity, computeCartography,
    computePartnerRankIssues, computePairDistanceTrend, computeAlignmentNeighborhoodImpurity,
} from '../utils/suspectSignals';

const CLASSIFICATION_SUSPECT_OPTIONS = [
    { value: 'confidently_wrong', label: 'Confidently wrong' },
    { value: 'oscillation', label: 'Oscillating' },
    { value: 'hd_impurity', label: 'HD neighborhood impurity' },
    { value: 'cartography_hard', label: 'Cartography: hard-to-learn' },
    { value: 'cartography_ambiguous', label: 'Cartography: ambiguous' },
];
const ALIGNMENT_SUSPECT_OPTIONS = [
    { value: 'partner_rank', label: 'Partner beyond top-k' },
    { value: 'pair_distance_trend', label: 'Diverging pairs' },
    { value: 'alignment_impurity', label: 'Neighborhood impurity' },
];
type SampleTag = {
    num: number;
    title: string;
}

interface LabelProps {
    label: string;
    colorArray: number[];
    onColorChange: (newColor: [number, number, number]) => void;
}

interface FunctionPanelProps {
    onUpdateProjection: () => Promise<void>;
    refineReady?: boolean;
    refineStatusMessage?: string | null;
}
const CompactCheckboxGroup = styled(Checkbox.Group)`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;

  .ant-checkbox-wrapper {
    display: flex;
    align-items: center;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background-color 0.2s ease;
    background-color: var(--surface-color, #ffffff);
    border: 1px solid var(--layout-border-color, #d9d9d9);

    &:hover {
      background-color: var(--token-active-color, #f5f5f5);
      border-color: var(--accent-blue, #3278F0);
    }
  }

  .ant-checkbox-checked .ant-checkbox-inner {
    background-color: var(--accent-blue, #3278F0);
    border-color: var(--accent-blue, #3278F0);
    width: 14px;
    height: 14px;
  }

  .ant-checkbox-inner {
    width: 14px;
    height: 14px;
  }
`;

const ColoredClassLabel: React.FC<LabelProps> = ({ label, colorArray, onColorChange }) => {
    const inputRef = useRef<HTMLInputElement>(null);


    function setColorPickerOpacity(value: number) {
        const colorPickerItem = inputRef.current;
        if (!colorPickerItem) return;
        if (value) {
            colorPickerItem.style.opacity = '1';
            colorPickerItem.style.pointerEvents = 'auto';
        } else {
            colorPickerItem.style.opacity = '0';
            colorPickerItem.style.pointerEvents = 'none';
        }
    }

    return (
        <div
            className="class-item"
            key={label}
            onMouseOver={() => setColorPickerOpacity(1)}
            onMouseLeave={() => setColorPickerOpacity(0)}
        >
            <input
                type="color"
                value={rgbArrToHex(colorArray)}
                onChange={(e) => onColorChange(hexToRgbArray((e.target as HTMLInputElement).value))}
            />
            <span style={{ color: rgbArrToHex(colorArray) }}>
                {label}
            </span>
        </div>
    )
}

function rgbArrToHex(rgbArray: number[]) {
    return '#' + rgbArray.map(c => c.toString(16).padStart(2, '0')).join('');
}

function hexToRgbArray(hex: string): [number, number, number] {
    hex = hex.replace(/^#/, '');
    const bigint = parseInt(hex, 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return [r, g, b];
}


function LensLegendDot({ color, label }: { color: string; label: string }) {
    return (
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
            {label}
        </span>
    );
}

export function FunctionPanel({ onUpdateProjection, refineReady = true, refineStatusMessage = null }: FunctionPanelProps) {
    const { tokenList, labelDict, colorDict, setColorDict, selectedIndices, setSelectedIndices, setShownData, pointSize, setPointSize, mode, setMode, epoch, allEpochData } =
        useDefaultStore(["tokenList","labelDict", "colorDict", "setColorDict", "selectedIndices", "setSelectedIndices", "setShownData", "pointSize", "setPointSize", "mode", "setMode", "epoch", "allEpochData"]);
    const { refineMetrics } = useDefaultStore(['refineMetrics']);
    const { revealOriginalNeighbors, revealProjectionNeighbors, setRevealOriginalNeighbors, setRevealProjectionNeighbors } =
        useDefaultStore(["revealOriginalNeighbors", "revealProjectionNeighbors", "setRevealOriginalNeighbors", "setRevealProjectionNeighbors"]);
    const { showIndex, showLabel, showBackground, showTrail, setShowIndex, setShowLabel, setShowBackground, setShowTrail } =
        useDefaultStore(["showIndex","showLabel","showBackground","showTrail","setShowIndex","setShowLabel","setShowBackground","setShowTrail"]);
    const { focusMode, setFocusMode } = useDefaultStore(['focusMode', 'setFocusMode']);
    const { boxSelectActive, setBoxSelectActive, refineFocusType, setRefineFocusType, secondaryIndices, setSecondaryIndices, setSecondaryBoxes } =
        useDefaultStore(['boxSelectActive', 'setBoxSelectActive', 'refineFocusType', 'setRefineFocusType', 'secondaryIndices', 'setSecondaryIndices', 'setSecondaryBoxes']);
    const { neighborDisplayIndices, setNeighborDisplayIndices } =
        useDefaultStore(['neighborDisplayIndices', 'setNeighborDisplayIndices']);
    const { refineTopK, setRefineTopK } =
        useDefaultStore(['refineTopK', 'setRefineTopK']);
    const { refinePriority, setRefinePriority } =
        useDefaultStore(['refinePriority', 'setRefinePriority']);
    const { showPreRefine, setShowPreRefine } =
        useDefaultStore(['showPreRefine', 'setShowPreRefine']);
    const { contentPath, vis_method, visID, setValue } =
        useDefaultStore(['contentPath', 'vis_method', 'visID', 'setValue']);
    const { inherentLabelData, setHoveredIndex } =
        useDefaultStore(['inherentLabelData', 'setHoveredIndex']);
    const { distortionLensOn, setDistortionLensOn } =
        useDefaultStore(['distortionLensOn', 'setDistortionLensOn']);
    const { activeRefineSessionId } =
        useDefaultStore(['activeRefineSessionId']);
    const { refineStatus } = useDefaultStore(['refineStatus']);
    const { showRefineTrails, setShowRefineTrails } =
        useDefaultStore(['showRefineTrails', 'setShowRefineTrails']);
    const { refineSessions } = useDefaultStore(['refineSessions']);
    const { taskType, alignment } = useDefaultStore(['taskType', 'alignment']);

    // B3 Undo: discard the refined result on the backend (graceful fallback then
    // serves the baseline) and revert the current epoch's displayed projection.
    const handleResetRefine = async () => {
        const ed = allEpochData[epoch];
        try {
            await BackendAPI.discardRefine(contentPath, vis_method, visID);
        } catch (e) {
            console.error('discardRefine failed', e);
        }
        if (ed?.originalProjection) {
            setValue('allEpochData', { ...allEpochData, [epoch]: { ...ed, projection: ed.originalProjection } });
        }
        setShowPreRefine(false);
        setShowRefineTrails(false);      // K: nothing to point at after a revert
        setValue('refineMetrics', null);
        setValue('refinedEpochs', []);   // C2: nothing refined after a full revert
        message.success('Refinement reverted to baseline.');
    };
    
    useEffect(() => {
        if (pointSize < 1) setPointSize(1);
        else if (pointSize > 5) setPointSize(5);
    }, [pointSize, setPointSize]);

    // Keep neighborDisplayIndices in sync with selectedIndices:
    // single selection → auto-show; multiple → keep intersection, default to first
    useEffect(() => {
        if (selectedIndices.length === 0) {
            setNeighborDisplayIndices([]);
        } else if (selectedIndices.length === 1) {
            setNeighborDisplayIndices([selectedIndices[0]]);
        } else {
            const sel = new Set(selectedIndices);
            const kept = neighborDisplayIndices.filter(i => sel.has(i));
            setNeighborDisplayIndices(kept.length > 0 ? kept : [selectedIndices[0]]);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedIndices]);

    const pointSizeMarks: Record<number, string> = { 1: '1', 2: '2', 3: '3', 4: '4', 5: '5' };
    const pointSizeLabel = pointSizeMarks[pointSize] ?? pointSize.toString();

    function changeLabelColor(i: number, newColor: [number, number, number]) {
        setColorDict(new Map([...colorDict, [i, newColor]]));
    }

    // J: progressive disclosure — refine expert params live in a collapsed
    // "Advanced" group inside Precision Control, off the default path.
    const [advancedOpen, setAdvancedOpen] = useState(false);

    // Suspect Samples: task-aware data-quality diagnostics, a second lens
    // alongside the projection-focused Distortion Lens.
    const [suspectSignal, setSuspectSignal] = useState<string>('confidently_wrong');
    // Accepts both the extension's enum value ('Code-Retrieval') and the
    // string real EIF-jump sessions actually send ('Alignment') — the two
    // callers never agreed on one name.
    const isAlignmentTask = taskType === 'Code-Retrieval' || taskType === 'Alignment';
    const suspectSignalOptions = isAlignmentTask ? ALIGNMENT_SUSPECT_OPTIONS : CLASSIFICATION_SUSPECT_OPTIONS;
    const effectiveSuspectSignal = suspectSignalOptions.some((o) => o.value === suspectSignal)
        ? suspectSignal
        : suspectSignalOptions[0].value;

    // NOTE always add state as middle dependency
    const [searchValue, setSearchValue] = useState('');
    const { tokenList: searchFromOptions } = useDefaultStore(['tokenList']);

    const limitOfHistory = 5;
    const [searchHistory, setSearchHistory] = useState<string[]>([]);
    const searchHistoryFiltered = searchHistory.filter((item) => item.includes(searchValue));
    const [searchHistoryOpen, setSearchHistoryOpen] = useState(false);
    const searchElementRef = useRef<RefSelectProps>(null);

    const [allSearchResult, setAllSearchResult] = useState<SampleTag[]>([]);

    const searchFrom = (text: string, items: SampleTag[], limit: number | null = 3) => {
        const res: SampleTag[] = [];

        let cnt = 0;

        for (const item of items) {
            if (item.title.toLowerCase().includes(text.toLowerCase())) {
                if (limit !== null && cnt >= limit) {
                    return res;
                }
                res.push(item);
                cnt++;
            }
        }

        return res;
    }

    const handleSearch = (text: string, byEnter: boolean = false) => {
        if (text === searchValue) return;
        setSearchValue(text);

        // prevent searching all
        if (!text) {
            setAllSearchResult([]);
            setSearchHistoryOpen(false);
            return;
        }

        setSearchHistoryOpen(true);

        const res = searchFrom(text, searchFromOptions.map((v, i) => {
            return {
                num: i,
                title: v
            }
        }), null);
        setAllSearchResult(res);

        if (byEnter) {
            addHistory(text);
        }
    };
    const addHistory = (text: string) => {
        if (!text) return;

        const nonDuplicateHistory = searchHistory.filter((item) => item !== text);
        setSearchHistory([text, ...nonDuplicateHistory].slice(0, limitOfHistory));
    }
    const renderSearchHistoryOption = (text: string) => {
        return {
            value: text,
            label: text
        }
    }
    const searchHistoryRender = (history: string[]) => {
        return history.map(renderSearchHistoryOption);
    }

    const searchResultRender = (item: SampleTag) => {
        return (
            <List.Item
                key={item.num}
                className={"search-result-sample" + (selectedIndices.includes(item.num) ? ' locked' : '')}
                onClick={() => {
                    const newSelectedIndices = selectedIndices.includes(item.num)
                        ? selectedIndices.filter(i => i !== item.num)
                        : [...selectedIndices, item.num];

                    setSelectedIndices(newSelectedIndices);
                }}
            >
                <div className="search-result-sample-field">
                    <span className="field-tag tag-1">index</span>
                    <span className="field-value">{item.num}</span>
                </div>
                <div className="search-result-sample-field">
                    <span className="field-tag tag-2">text</span>
                    <span className="field-value">{item.title}</span>
                </div>
            </List.Item>
        )
    }

    const [selectedItems, setSelectedItems] = useState<SampleTag[]>([]);

    const handleClose = (item: SampleTag) => {
        const newSelectedIndices = selectedIndices.filter(i => i !== item.num);
        setSelectedIndices(newSelectedIndices);
    };

    useEffect(() => {
        setSelectedItems(Array.from(selectedIndices).map((num) => ({
            num,
            title: tokenList ? tokenList[num] ?? '' : ''
        })));
    }, [selectedIndices, tokenList]);

    const selectedRelations = useMemo(() => {
        const epochData = allEpochData[epoch];
        if (!epochData || selectedIndices.length < 2) return [];

        const relations: Array<{
            key: string;
            left: SampleTag;
            right: SampleTag;
            hdMutual: boolean;
            hdOneWay: boolean;
            ldMutual: boolean;
            ldOneWay: boolean;
        }> = [];

        for (let i = 0; i < selectedItems.length; i++) {
            for (let j = i + 1; j < selectedItems.length; j++) {
                const left = selectedItems[i];
                const right = selectedItems[j];
                const leftHd = epochData.originalNeighbors?.[left.num]?.includes(right.num) ?? false;
                const rightHd = epochData.originalNeighbors?.[right.num]?.includes(left.num) ?? false;
                const leftLd = epochData.projectionNeighbors?.[left.num]?.includes(right.num) ?? false;
                const rightLd = epochData.projectionNeighbors?.[right.num]?.includes(left.num) ?? false;

                relations.push({
                    key: `${left.num}-${right.num}`,
                    left,
                    right,
                    hdMutual: leftHd && rightHd,
                    hdOneWay: (leftHd || rightHd) && !(leftHd && rightHd),
                    ldMutual: leftLd && rightLd,
                    ldOneWay: (leftLd || rightLd) && !(leftLd && rightLd),
                });
            }
        }

        return relations;
    }, [allEpochData, epoch, selectedIndices, selectedItems]);

    // Focus Neighbors: HD top-k of the primary selected point, each row marked
    // satisfied (currently also an LD top-k neighbor) or not — a text-legible
    // complement to the red/blue rings drawn on the canvas.
    const focusNeighborRows = useMemo(() => {
        const epochData = allEpochData[epoch];
        if (!epochData || selectedIndices.length === 0) return [];
        const focusId = selectedIndices[0];
        const focusPos = rawIndexToProjectionPosition(focusId, epochData.indexList);
        const hdPositions = (epochData.originalNeighbors?.[focusPos] ?? []).slice(0, refineTopK);
        const hdIds = convertNeighborPositionsToRawIndices(hdPositions, epochData.indexList);
        const ldPositions = computeProjectionNeighborPositionsForPoint(focusId, epochData.projection, epochData.indexList, refineTopK);
        const ldIdSet = new Set(convertNeighborPositionsToRawIndices(ldPositions, epochData.indexList));

        return hdIds.map((id, i) => ({
            id,
            hdRank: i + 1,
            satisfied: ldIdSet.has(id),
            label: tokenList?.[id] ?? labelDict.get(inherentLabelData[id]) ?? '',
        }));
    }, [allEpochData, epoch, selectedIndices, refineTopK, tokenList, labelDict, inherentLabelData]);

    const focusNeighborSatisfiedCount = focusNeighborRows.filter((row) => row.satisfied).length;

    // B: Most Distorted — global (not view-filtered) recommendation list so
    // users have somewhere to start instead of hunting for a bad point by eye.
    const mostDistortedPoints = useMemo(() => {
        const epochData = allEpochData[epoch];
        const npMap = computeAllPointsNeighborPreservation(
            epochData?.originalNeighbors,
            epochData?.projectionNeighbors,
            epochData?.indexList,
            refineTopK,
        );
        return Array.from(npMap.entries())
            .sort((a, b) => a[1] - b[1])
            .slice(0, 10)
            .map(([id, np]) => ({ id, np, label: tokenList?.[id] ?? labelDict.get(inherentLabelData[id]) ?? '' }));
    }, [allEpochData, epoch, refineTopK, tokenList, labelDict, inherentLabelData]);

    // C: session history helpers — a per-run outcome label plus JSON download
    // and Markdown-table copy for reports.
    const fmtNp = (v: number | null) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`);
    const sessionOutcome = (r: typeof refineSessions[number]) =>
        r.stoppedByUser ? 'stopped' : r.converged === true ? 'converged' : r.converged === false ? (r.exitReason ?? 'partial') : 'done';

    const handleExportSessionsJson = () => {
        const blob = new Blob([JSON.stringify(refineSessions, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `refine-sessions-${new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-')}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const handleCopySessionsMarkdown = async () => {
        const header = '| Time | Epoch | Focus | k | Priority | NP before → after | Duration | Outcome |\n|---|---|---|---|---|---|---|---|';
        const rows = refineSessions.map((r) => {
            const focus = `#${r.focusIds[0]}${r.focusIds.length > 1 ? ` +${r.focusIds.length - 1}` : ''}`;
            return `| ${new Date(r.timestamp).toLocaleTimeString()} | ${r.epoch} | ${focus} | ${r.topK} | ${r.priority} | ${fmtNp(r.npBefore)} → ${fmtNp(r.npAfter)} | ${(r.durationMs / 1000).toFixed(0)}s | ${sessionOutcome(r)} |`;
        });
        try {
            await navigator.clipboard.writeText([header, ...rows].join('\n'));
            message.success('History copied as Markdown.');
        } catch {
            message.error('Clipboard unavailable — use Export JSON instead.');
        }
    };

    type SuspectDisplayRow = { key: string; title: string; detail: string; ids: number[] };

    // Live refine polling replaces the whole `allEpochData` object every ~200ms
    // even though it only ever patches `projection`/`projectionNeighbors` for
    // the epoch being refined — prediction/predProbability/originalNeighbors
    // (everything the classification and HD-based alignment signals read)
    // never change from refine. Depending on this epoch-keys fingerprint
    // (a primitive string, stable across those in-place patches) instead of
    // the `allEpochData` object reference itself avoids re-running the O(E·N)
    // Cartography/softmax scan on every refine tick. The one signal that does
    // read live LD projections (pair distance trend) intentionally only
    // refreshes on natural triggers (epoch switch, new epoch load, signal
    // change) rather than live during an in-progress refine — it's a
    // full-run convergence check, not a live refine monitor.
    const epochKeysFingerprint = Object.keys(allEpochData).sort((a, b) => Number(a) - Number(b)).join(',');

    // Suspect Samples: dispatch to the selected signal and format a uniform,
    // explainable row (title + why-flagged detail) — never a bare score.
    const suspectRows = useMemo<SuspectDisplayRow[]>(() => {
        const epochData = allEpochData[epoch];
        const label = (id: number) => tokenList?.[id] ?? labelDict.get(inherentLabelData[id]) ?? '';
        const pairTitle = (ids: [number, number]) => `#${ids[0]} ↔ #${ids[1]}`;

        if (isAlignmentTask) {
            switch (effectiveSuspectSignal) {
                case 'pair_distance_trend':
                    return computePairDistanceTrend(alignment, allEpochData).slice(0, 10).map((r) => ({
                        key: pairTitle(r.pairIds), title: pairTitle(r.pairIds), ids: r.pairIds,
                        detail: `LD dist ${r.firstDist.toFixed(2)} → ${r.lastDist.toFixed(2)} (${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(2)})`,
                    }));
                case 'alignment_impurity':
                    return computeAlignmentNeighborhoodImpurity(alignment, epochData, refineTopK).slice(0, 10).map((r) => ({
                        key: pairTitle(r.pairIds), title: pairTitle(r.pairIds), ids: r.pairIds,
                        detail: `${(r.purity * 100).toFixed(0)}% of HD neighbors share this cluster`,
                    }));
                case 'partner_rank':
                default:
                    return computePartnerRankIssues(alignment, epochData, refineTopK).slice(0, 10).map((r) => ({
                        key: pairTitle(r.pairIds), title: pairTitle(r.pairIds), ids: r.pairIds,
                        detail: r.withinTopK ? `partner rank ${r.rank} of top-${refineTopK}` : `partner beyond top-${refineTopK}`,
                    }));
            }
        }

        switch (effectiveSuspectSignal) {
            case 'oscillation':
                return computeOscillation(allEpochData).slice(0, 10).map((r) => ({
                    key: String(r.id), title: `#${r.id} ${label(r.id)}`, ids: [r.id],
                    detail: `flipped ${r.flips}× across ${r.epochsSeen} epochs`,
                }));
            case 'hd_impurity':
                return computeHdImpurity(epochData, inherentLabelData, refineTopK).slice(0, 10).map((r) => ({
                    key: String(r.id), title: `#${r.id} ${label(r.id)}`, ids: [r.id],
                    detail: `${r.sameLabelCount}/${r.total} HD neighbors share this label`,
                }));
            case 'cartography_hard':
                return computeCartography(allEpochData, inherentLabelData)
                    .sort((a, b) => a.confidence - b.confidence).slice(0, 10).map((r) => ({
                        key: String(r.id), title: `#${r.id} ${label(r.id)}`, ids: [r.id],
                        detail: `conf ${(r.confidence * 100).toFixed(0)}% · var ${(r.variability * 100).toFixed(0)}% · correct ${(r.correctness * 100).toFixed(0)}%`,
                    }));
            case 'cartography_ambiguous':
                return computeCartography(allEpochData, inherentLabelData)
                    .sort((a, b) => b.variability - a.variability).slice(0, 10).map((r) => ({
                        key: String(r.id), title: `#${r.id} ${label(r.id)}`, ids: [r.id],
                        detail: `var ${(r.variability * 100).toFixed(0)}% · conf ${(r.confidence * 100).toFixed(0)}% · correct ${(r.correctness * 100).toFixed(0)}%`,
                    }));
            case 'confidently_wrong':
            default:
                return computeConfidentlyWrong(epochData, inherentLabelData).slice(0, 10).map((r) => ({
                    key: String(r.id), title: `#${r.id} ${label(r.id)}`, ids: [r.id],
                    detail: `conf ${(r.confidence * 100).toFixed(0)}% → predicted "${labelDict.get(r.predictedLabel) ?? r.predictedLabel}" (true: "${labelDict.get(r.trueLabel) ?? r.trueLabel}")`,
                }));
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isAlignmentTask, effectiveSuspectSignal, epochKeysFingerprint, epoch, alignment, refineTopK, inherentLabelData, labelDict, tokenList]);

    // D: stop a running refine early and keep its current (intermediate) result
    // instead of waiting out the full 90s/300s budget.
    const handleStopRefine = async () => {
        if (!activeRefineSessionId) return;
        try {
            await BackendAPI.stopRefineSession(activeRefineSessionId);
            message.info('Stopping refine — keeping current result...');
        } catch (e) {
            console.error('stopRefineSession failed', e);
            message.error('Failed to stop refine.');
        }
    };

    return (
        <div className="info-column">
            <FunctionalBlock label="Search">
                <AutoComplete
                    style={{ width: '100%', paddingRight: '0.4em'}} // Set width to 100% for responsiveness
                    ref={searchElementRef}
                    options={searchHistoryRender(searchHistoryFiltered)}
                    value={searchValue}
                    open={searchHistoryOpen}
                    onChange={(value: string) => { handleSearch(value) }}
                    onBlur={() => {
                        addHistory(searchValue);    // TODO only add successful history
                        setSearchHistoryOpen(false);
                    }}
                    onFocus={() => handleSearch(searchValue)}
                    onKeyDown={(e: { key: string; }) => {
                        if (e.key === 'Enter') {
                            handleSearch(searchValue, true);
                            setSearchHistoryOpen(false);
                        } else if (e.key === 'Escape') {
                            searchElementRef.current?.blur();
                        }
                    }}
                    onSelect={() => {
                        searchElementRef.current?.blur();
                    }}
                    onClear={() => {
                        setSearchHistoryOpen(false);
                    }}
                    defaultActiveFirstOption={false}
                    notFoundContent={<div className='alt-text placeholder-block'>No item found</div>}
                    allowClear
                >
                    <Input onClick={() => {
                        setSearchHistoryOpen(true);
                    }} />
                </AutoComplete>
                {
                    (allSearchResult.length > 0 || searchValue !== '')
                    &&
                    <ComponentBlock label="Search Result">
                        {
                            allSearchResult.length > 0
                                ?
                                (
                                    <List className="search-result"
                                        size="small"
                                        bordered
                                        dataSource={allSearchResult}
                                        renderItem={searchResultRender}
                                    />
                                )
                                :
                                (searchValue && <div className='alt-text placeholder-block'>No item found</div>)
                        }
                    </ComponentBlock>
                }
            </FunctionalBlock>
<FunctionalBlock label="Precision Control">
    {/* Row: Focus Type */}
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Focus Type</span>
        <Select
            size="small"
            value={refineFocusType}
            style={{ width: 110 }}
            onChange={(v: RefineFocusType) => {
                setRefineFocusType(v);
                if (v === 'uniform') { setSecondaryIndices([]); setSecondaryBoxes([]); }
            }}
            options={[
                { value: 'uniform', label: 'All Focus' },
                { value: 'tiered', label: 'Tiered' },
            ]}
        />
    </div>

    {/* Box Select button */}
    <Button
        size="small"
        block
        icon={boxSelectActive ? <MousePointer2 size={12} /> : <BoxSelect size={12} />}
        onClick={() => setBoxSelectActive(!boxSelectActive)}
        style={{
            marginBottom: 8,
            background: boxSelectActive ? 'var(--accent-blue)' : undefined,
            color: boxSelectActive ? '#fff' : undefined,
            borderColor: boxSelectActive ? 'var(--accent-blue)' : undefined,
            fontWeight: boxSelectActive ? 600 : undefined,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 4,
        }}
    >
        {boxSelectActive ? 'Box Select: ON — click to exit' : 'Start Box Select'}
    </Button>

    {/* Point counts */}
    {refineFocusType === 'tiered' ? (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: 6 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', border: '2px solid var(--color-warning)', flexShrink: 0 }} />
                <span><b style={{ color: 'var(--text-primary)' }}>{selectedIndices.length}</b> primary</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', border: '1.5px dashed var(--accent-blue)', flexShrink: 0 }} />
                <span><b style={{ color: 'var(--text-primary)' }}>{secondaryIndices.length}</b> secondary</span>
                {secondaryIndices.length > 0 && (
                    <Button size="small" icon={<X size={10} />} style={{ padding: '0 4px', height: 16, fontSize: 10, marginLeft: 'auto' }} danger
                        onClick={() => { setSecondaryIndices([]); setSecondaryBoxes([]); }}>
                        Clear
                    </Button>
                )}
            </div>
        </div>
    ) : (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-blue)', marginRight: 6 }} />
            <b style={{ color: 'var(--text-primary)' }}>{selectedIndices.length}</b> points selected
        </div>
    )}

    {/* Update button */}
    <Button
        type="primary"
        block
        size="small"
        icon={<SyncOutlined />}
        disabled={!refineReady}
        onClick={onUpdateProjection}
        style={{ borderRadius: 4, fontWeight: 500, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}
    >
        {refineReady ? 'Update Projection' : 'Preparing...'}
    </Button>
    {!refineReady && refineStatusMessage && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.4, marginTop: 6 }}>
            {refineStatusMessage}
        </div>
    )}
    {/* D: stop a running refine early and keep the current intermediate result */}
    {refineStatus === 'running' && activeRefineSessionId && (
        <Button
            size="small"
            block
            icon={<X size={12} />}
            onClick={handleStopRefine}
            style={{ marginTop: 6, borderRadius: 4 }}
        >
            Stop &amp; keep current result
        </Button>
    )}

    {/* J: expert refine params, collapsed by default */}
    <div
        onClick={() => setAdvancedOpen(!advancedOpen)}
        style={{ marginTop: 10, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-muted)', cursor: 'pointer', userSelect: 'none' }}
    >
        {advancedOpen ? '▾' : '▸'} Advanced
    </div>
    {advancedOpen && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 6 }}>
            {/* C3: neighborhood size (top-k) the refine objective preserves */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Tooltip title="Number of nearest neighbors refine tries to align (HD top-k = LD top-k). Range 3–20.">
                    <span style={{ minWidth: 80, fontSize: 11, color: 'var(--text-muted)' }}>Refine top-k</span>
                </Tooltip>
                <InputNumber
                    size="small" style={{ flex: 1 }}
                    min={3} max={20} step={1} precision={0}
                    value={refineTopK}
                    onChange={(v) => {
                        if (typeof v === 'number' && !Number.isNaN(v)) {
                            setRefineTopK(Math.max(3, Math.min(20, Math.round(v))));
                        }
                    }}
                />
            </div>
            {/* B1: accuracy ↔ layout tradeoff */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Tooltip title="Preserve layout: keep the current arrangement (may accept <100% neighbor accuracy). Max accuracy: pull neighbors in aggressively, allowing more layout distortion.">
                    <span style={{ minWidth: 80, fontSize: 11, color: 'var(--text-muted)' }}>Refine goal</span>
                </Tooltip>
                <Select
                    size="small" style={{ flex: 1 }}
                    value={refinePriority <= 0.3 ? 'layout' : refinePriority >= 0.7 ? 'accuracy' : 'balanced'}
                    onChange={(v) => {
                        setRefinePriority(v === 'layout' ? 0.15 : v === 'accuracy' ? 0.9 : 0.5);
                    }}
                    options={[
                        { label: 'Preserve layout', value: 'layout' },
                        { label: 'Balanced', value: 'balanced' },
                        { label: 'Max accuracy', value: 'accuracy' },
                    ]}
                />
            </div>
        </div>
    )}
</FunctionalBlock>
            {selectedIndices.length > 0 && (
                <FunctionalBlock label="Focus Neighbors">
                    {focusNeighborRows.length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                                    HD top-{refineTopK} of #{selectedIndices[0]}
                                </span>
                                <span style={{
                                    fontSize: 11, fontWeight: 600,
                                    color: focusNeighborSatisfiedCount === focusNeighborRows.length ? 'var(--color-success)' : 'var(--text-primary)',
                                }}>
                                    {focusNeighborSatisfiedCount}/{focusNeighborRows.length} in LD
                                </span>
                            </div>
                            <div style={{ maxHeight: 180, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 2 }}>
                                {focusNeighborRows.map((row) => (
                                    <div
                                        key={row.id}
                                        onMouseEnter={() => setHoveredIndex(row.id)}
                                        onMouseLeave={() => setHoveredIndex(undefined)}
                                        style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '2px 4px', fontSize: 11 }}
                                    >
                                        <span style={{
                                            display: 'inline-block', width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                                            background: row.satisfied ? 'var(--color-success)' : 'var(--color-error)',
                                        }} />
                                        <span style={{ color: 'var(--text-muted)', minWidth: 16 }}>{row.hdRank}</span>
                                        <span style={{ color: 'var(--text-primary)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                            #{row.id}{row.label ? ` ${row.label}` : ''}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', padding: '10px 0' }}>
                            No HD neighbor data for this point yet.
                        </div>
                    )}
                </FunctionalBlock>
            )}
            <FunctionalBlock label="Refine Quality">
                {refineMetrics ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                        {/* ── POSITION group ── */}
                        <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 4 }}>
                            Position
                        </div>
                        {[
                            { label: 'Displacement', value: refineMetrics.focusDisplacement, format: (v: number) => v.toFixed(4), tip: 'Average 2D movement of focus points. Larger = more effect.' },
                            { label: 'Global Drift', value: refineMetrics.globalDrift, format: (v: number) => v.toFixed(4), tip: 'Average 2D movement of non-focus points. Smaller = more stable.' },
                        ].map(({ label, value, format, tip }) => (
                            <Tooltip key={label} title={tip} placement="left">
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'help', marginBottom: 3 }}>
                                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</span>
                                    <span style={{ fontSize: 11, fontWeight: 600, fontFamily: 'var(--metric-value-font)', color: value == null ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                                        {value == null ? '—' : format(value)}
                                    </span>
                                </div>
                            </Tooltip>
                        ))}

                        {/* ── STRUCTURE group ── */}
                        <div style={{ borderTop: '1px solid var(--layout-border-color)', margin: '6px 0 4px' }} />
                        <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 4 }}>
                            Structure
                        </div>
                        {[
                            {
                                label: 'NP (k=10)', value: refineMetrics.neighborPreservation,
                                format: (v: number) => `${(v * 100).toFixed(1)}%`,
                                dot: (v: number) => v * 100 >= 25 ? 'var(--color-success)' : v * 100 >= 10 ? 'var(--color-warning)' : 'var(--color-error)',
                                tip: 'Neighbor Preservation: fraction of HD top-10 neighbors in LD top-10. Range 5–40%. Higher is better.',
                            },
                            {
                                label: 'HD Rank', value: refineMetrics.meanRankHD,
                                format: (v: number) => v.toFixed(1),
                                dot: (v: number) => v <= 10 ? 'var(--color-success)' : v <= 25 ? 'var(--color-warning)' : 'var(--color-error)',
                                tip: 'Mean LD rank of HD top-10 neighbors. Lower is better. Ideal ≈ 5.5.',
                            },
                            {
                                label: 'Trustworthiness', value: refineMetrics.trustworthiness,
                                format: (v: number) => `${(v * 100).toFixed(1)}%`,
                                dot: (v: number) => v * 100 >= 85 ? 'var(--color-success)' : v * 100 >= 70 ? 'var(--color-warning)' : 'var(--color-error)',
                                tip: 'Are LD neighbors trustworthy? Penalises false LD neighbors. Satisfying: >70%. Excellent: >85%.',
                            },
                            {
                                label: 'Continuity', value: refineMetrics.continuity,
                                format: (v: number) => `${(v * 100).toFixed(1)}%`,
                                dot: (v: number) => v * 100 >= 85 ? 'var(--color-success)' : v * 100 >= 70 ? 'var(--color-warning)' : 'var(--color-error)',
                                tip: 'Are HD neighbors preserved in LD? Penalises missing HD neighbors in projection. Satisfying: >70%. Excellent: >85%.',
                            },
                        ].map(({ label, value, format, dot, tip }) => (
                            <Tooltip key={label} title={tip} placement="left">
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'help', marginBottom: 3 }}>
                                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</span>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                        <span style={{ fontSize: 11, fontWeight: 600, fontFamily: 'var(--metric-value-font)', color: value == null ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                                            {value == null ? '—' : format(value)}
                                        </span>
                                        {value != null && (
                                            <span style={{ display: 'inline-block', width: 7, height: 7, borderRadius: '50%', background: dot(value), flexShrink: 0 }} />
                                        )}
                                    </div>
                                </div>
                            </Tooltip>
                        ))}

                        {/* ── Focus / Drift ratio bar ── */}
                        <div style={{ borderTop: '1px solid var(--layout-border-color)', marginTop: 6, paddingTop: 6 }}>
                            <div style={{ height: 5, background: 'var(--layout-border-color)', borderRadius: 3, overflow: 'hidden' }}>
                                <div style={{
                                    height: '100%',
                                    width: `${Math.min(100, refineMetrics.globalDrift > 0
                                        ? Math.min(refineMetrics.focusDisplacement / (refineMetrics.focusDisplacement + refineMetrics.globalDrift) * 100, 100)
                                        : 100)}%`,
                                    background: 'linear-gradient(90deg, var(--color-success), var(--accent-blue))',
                                    borderRadius: 3,
                                    transition: 'width 0.4s ease',
                                }} />
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                                <span>focus</span><span>drift</span>
                            </div>
                        </div>

                        {/* B3: non-destructive before/after toggle */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 6 }}>
                            <Tooltip title="Show the pre-refine baseline layout. Toggle to compare before vs after — nothing is discarded.">
                                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Show baseline (before)</span>
                            </Tooltip>
                            <Switch
                                size="small"
                                checked={showPreRefine}
                                onChange={(v) => setShowPreRefine(v)}
                            />
                        </div>

                        {/* K: static baseline→refined displacement arrows on the canvas */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 6 }}>
                            <Tooltip title="Draw arrows from each cluster member's pre-refine position to its current one — a static record of what refine moved.">
                                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Show displacement</span>
                            </Tooltip>
                            <Switch
                                size="small"
                                checked={showRefineTrails}
                                onChange={(v) => setShowRefineTrails(v)}
                            />
                        </div>

                        {/* B3: undo — revert the refinement back to baseline */}
                        <Button
                            size="small"
                            danger
                            icon={<RefreshCw size={12} />}
                            style={{ marginTop: 6, width: '100%' }}
                            onClick={handleResetRefine}
                        >
                            Reset refine (revert to baseline)
                        </Button>
                    </div>
                ) : (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', padding: '10px 0' }}>
                        Run Update Projection to see metrics
                    </div>
                )}
            </FunctionalBlock>
            <FunctionalBlock label="Distortion Lens">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Tooltip title="Color every point by neighbor preservation (HD top-k ∩ LD top-k). Reveals where the projection is unfaithful, before you refine anything.">
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Show on canvas</span>
                    </Tooltip>
                    <Switch size="small" checked={distortionLensOn} onChange={setDistortionLensOn} />
                </div>
                {distortionLensOn && (
                    <div style={{ display: 'flex', gap: 12, marginBottom: 10, fontSize: 10, color: 'var(--text-muted)' }}>
                        <LensLegendDot color="#22c55e" label="≥25%" />
                        <LensLegendDot color="#f59e0b" label="≥10%" />
                        <LensLegendDot color="#ef4444" label="<10%" />
                    </div>
                )}
                <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 4 }}>
                    Most Distorted
                </div>
                {mostDistortedPoints.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        {mostDistortedPoints.map((row) => (
                            <div
                                key={row.id}
                                onClick={() => { setSelectedIndices([row.id]); }}
                                style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, padding: '3px 4px', borderRadius: 3, cursor: 'pointer', fontSize: 11 }}
                            >
                                <span style={{ color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    #{row.id}{row.label ? ` ${row.label}` : ''}
                                </span>
                                <span style={{ fontWeight: 600, color: row.np * 100 >= 10 ? 'var(--color-warning)' : 'var(--color-error)' }}>
                                    {(row.np * 100).toFixed(0)}%
                                </span>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', padding: '6px 0' }}>
                        No neighbor data loaded yet.
                    </div>
                )}
            </FunctionalBlock>
            <FunctionalBlock label="History" defaultCollapsed={true}>
                {refineSessions.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
                            <Button size="small" style={{ flex: 1, fontSize: 10 }} onClick={handleExportSessionsJson}>
                                Export JSON
                            </Button>
                            <Button size="small" style={{ flex: 1, fontSize: 10 }} onClick={handleCopySessionsMarkdown}>
                                Copy Markdown
                            </Button>
                        </div>
                        <div style={{ maxHeight: 200, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 }}>
                            {[...refineSessions].reverse().map((r) => (
                                <div
                                    key={r.id + r.timestamp}
                                    onClick={() => {
                                        setValue('epoch', r.epoch);
                                        setSelectedIndices(r.focusIds);
                                    }}
                                    style={{
                                        display: 'flex', flexDirection: 'column', gap: 1,
                                        padding: '4px 6px', borderRadius: 4, cursor: 'pointer',
                                        border: '1px solid var(--layout-border-color)', fontSize: 11,
                                    }}
                                >
                                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6 }}>
                                        <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                                            Ep {r.epoch} · #{r.focusIds[0]}{r.focusIds.length > 1 ? ` +${r.focusIds.length - 1}` : ''}
                                        </span>
                                        <span style={{ color: 'var(--text-muted)' }}>
                                            {new Date(r.timestamp).toLocaleTimeString()}
                                        </span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 6, color: 'var(--text-muted)' }}>
                                        <span>
                                            NP {fmtNp(r.npBefore)} → <b style={{
                                                color: (r.npAfter ?? 0) >= (r.npBefore ?? 0) ? 'var(--color-success)' : 'var(--color-error)',
                                            }}>{fmtNp(r.npAfter)}</b>
                                        </span>
                                        <span>{(r.durationMs / 1000).toFixed(0)}s · {sessionOutcome(r)}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                ) : (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', padding: '6px 0' }}>
                        No refine runs yet this session.
                    </div>
                )}
            </FunctionalBlock>
            <FunctionalBlock label="Suspect Samples">
                <div style={{ marginBottom: 8 }}>
                    <Select
                        size="small"
                        style={{ width: '100%' }}
                        value={effectiveSuspectSignal}
                        onChange={(v) => setSuspectSignal(v)}
                        options={suspectSignalOptions}
                    />
                </div>
                {suspectRows.length > 0 ? (
                    <div style={{ maxHeight: 220, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {suspectRows.map((row) => (
                            <div
                                key={row.key}
                                onClick={() => setSelectedIndices(row.ids)}
                                style={{ padding: '4px 6px', borderRadius: 4, cursor: 'pointer', border: '1px solid var(--layout-border-color)', fontSize: 11 }}
                            >
                                <div style={{ color: 'var(--text-primary)', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    {row.title}
                                </div>
                                <div style={{ color: 'var(--text-muted)' }}>{row.detail}</div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', padding: '6px 0' }}>
                        {isAlignmentTask && alignment.length === 0
                            ? 'No alignment data found (dataset/align.json missing).'
                            : 'No suspects found for this lens yet.'}
                    </div>
                )}
            </FunctionalBlock>
            <FunctionalBlock label="Categories" defaultCollapsed={true}>
                <ComponentBlock>
                    <div className="class-list">
                        {
                            Array.from(labelDict.keys()).length
                                ?
                                Array.from(labelDict.keys()).map((labelNum) =>
                                    <ColoredClassLabel
                                        key={labelNum}
                                        label={labelDict.get(labelNum)!}
                                        colorArray={colorDict.get(labelNum)!}
                                        onColorChange={(newColor) => changeLabelColor(labelNum, newColor)}
                                    />
                                )
                                :
                                <div className='alt-text placeholder-block'>No class is determined</div>
                        }
                    </div>
                </ComponentBlock>
            </FunctionalBlock>
            <FunctionalBlock label="Selected">
                <ComponentBlock>
                    {selectedItems.length > 0 && (
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{selectedItems.length} selected</span>
                            <Button size="small" danger onClick={() => setSelectedIndices([])}>
                                Clear All
                            </Button>
                        </div>
                    )}
                    <div className="tag-list">
                        {
                            selectedItems.length
                                ?
                                selectedItems.map((item) => (
                                    <Tag className='sample-tag'
                                        closeIcon
                                        onClick={(e: { preventDefault: () => void; }) => {
                                            e.preventDefault();
                                            handleClose(item);
                                        }}
                                        onClose={(e: { preventDefault: () => void; }) => {
                                            e.preventDefault();
                                            handleClose(item);
                                        }}
                                        key={item.num}
                                    >
                                        {item.num}. {item.title}
                                    </Tag>
                                ))
                                :
                                <div className='alt-text placeholder-block'>No selected item</div>
                        }
                    </div>
                    {selectedRelations.length > 0 && (
                        <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
                            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>Relations</div>
                            {selectedRelations.map((relation) => (
                                <div key={relation.key} style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', fontSize: 11 }}>
                                    <span style={{ color: 'var(--text-primary)' }}>{relation.left.num}</span>
                                    <span style={{ color: 'var(--text-muted)' }}>↔</span>
                                    <span style={{ color: 'var(--text-primary)' }}>{relation.right.num}</span>
                                    {relation.hdMutual && <Tag color="red">HD↔</Tag>}
                                    {relation.hdOneWay && <Tag color="volcano">HD→</Tag>}
                                    {relation.ldMutual && <Tag color="blue">LD↔</Tag>}
                                    {relation.ldOneWay && <Tag color="geekblue">LD→</Tag>}
                                    {!relation.hdMutual && !relation.hdOneWay && !relation.ldMutual && !relation.ldOneWay && (
                                        <Tag>None</Tag>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </ComponentBlock>
            </FunctionalBlock>
            <FunctionalBlock label="Settings" defaultCollapsed={true}>
                <ComponentBlock>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                        {/* Point Size */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ minWidth: 80, fontSize: 11, color: 'var(--text-muted)' }}>Point Size</span>
                            <Slider
                                min={1} max={5} step={1} dots marks={pointSizeMarks}
                                value={pointSize}
                                onChange={(v) => setPointSize(v as number)}
                                style={{ flex: 1, minWidth: 60 }}
                            />
                        </div>

                        <div style={{ borderTop: '1px solid var(--layout-border-color)', margin: '2px 0' }} />

                        {/* Mode */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ minWidth: 80, fontSize: 11, color: 'var(--text-muted)' }}>Mode</span>
                            <Select
                                size="small" style={{ flex: 1 }} value={mode} onChange={(v) => setMode(v)}
                                options={[
                                    { label: 'Points', value: 'points' },
                                    { label: 'Density', value: 'density' },
                                ]}
                            />
                        </div>

                        {/* Neighbors */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ minWidth: 80, fontSize: 11, color: 'var(--text-muted)' }}>Neighbors</span>
                            <Select
                                size="small" style={{ flex: 1 }}
                                value={
                                    revealOriginalNeighbors && revealProjectionNeighbors ? 'both'
                                    : revealOriginalNeighbors ? 'original'
                                    : revealProjectionNeighbors ? 'projection' : 'none'
                                }
                                onChange={(v) => {
                                    setRevealOriginalNeighbors(v === 'original' || v === 'both');
                                    setRevealProjectionNeighbors(v === 'projection' || v === 'both');
                                }}
                                options={[
                                    { label: 'None', value: 'none' },
                                    { label: 'Original', value: 'original' },
                                    { label: 'Projection', value: 'projection' },
                                    { label: 'Both', value: 'both' },
                                ]}
                            />
                        </div>

                        {/* Show-neighbor checklist — only when multiple points selected */}
                        {selectedIndices.length > 1 && (revealOriginalNeighbors || revealProjectionNeighbors) && (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, paddingLeft: 4 }}>
                                <span style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 1 }}>
                                    Show neighbors for:
                                </span>
                                {selectedIndices.map((idx) => {
                                    const checked = neighborDisplayIndices.includes(idx);
                                    return (
                                        <label key={idx} style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer' }}>
                                            <input
                                                type="checkbox"
                                                checked={checked}
                                                onChange={(e) => {
                                                    if (e.target.checked) {
                                                        setNeighborDisplayIndices([...neighborDisplayIndices, idx]);
                                                    } else {
                                                        setNeighborDisplayIndices(neighborDisplayIndices.filter(i => i !== idx));
                                                    }
                                                }}
                                                style={{ width: 12, height: 12, accentColor: 'var(--accent-blue, #3278F0)', cursor: 'pointer' }}
                                            />
                                            <span style={{ fontSize: 11, color: 'var(--primary-text)' }}>
                                                #{idx}
                                            </span>
                                        </label>
                                    );
                                })}
                            </div>
                        )}

                        <div style={{ borderTop: '1px solid var(--layout-border-color)', margin: '2px 0' }} />

                        {/* Display switches — flat rows, no nested card */}
                        {[
                            { label: 'Show Label',      checked: showLabel,      onChange: setShowLabel },
                            { label: 'Show Index',      checked: showIndex,      onChange: setShowIndex },
                            { label: 'Show Trail',      checked: showTrail,      onChange: setShowTrail },
                            { label: 'Show Background', checked: showBackground, onChange: setShowBackground },
                        ].map(({ label, checked, onChange }) => (
                            <div key={label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</span>
                                <Switch size="small" checked={checked} onChange={(v) => onChange(v)} />
                            </div>
                        ))}
                    </div>
                </ComponentBlock>
            </FunctionalBlock>
            <FunctionalBlock label="Filter" defaultCollapsed={true}>
                <ComponentBlock>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                        <CompactCheckboxGroup
                            options={[
                                { label: 'Train Data', value: 'train' },
                                { label: 'Test Data', value: 'test' },
                            ]}
                            defaultValue={['train', 'test']}
                            onChange={(checkedValues) => {
                                setShownData(checkedValues as string[]);
                            }}
                        />
                    </div>
                </ComponentBlock>
            </FunctionalBlock>
            <FunctionalBlock label="Highlight" defaultCollapsed={true}>
                <HighlightOptionBlock />
            </FunctionalBlock>
        </div>
    )
}

export default FunctionPanel;


function HighlightOptionBlock() {
    const { highlightData, setHighlightData } = useDefaultStore(["highlightData", "setHighlightData"]);

    const [highlightTypes, setHighlightTypes] = useState([
        { type: 'prediction_error', label: 'Prediction Error', enabled: false, icon: 'error', description: 'Samples with wrong prediction at current epoch.' },
        { type: 'prediction_flip', label: 'Prediction Flip', enabled: false, icon: 'flip', description: 'Samples with prediction flip at current epoch.' }
    ]);

    const handleToggleHighlightType = (type: string) => {
        const updatedhighlightTypes = highlightTypes.map(highlight => highlight.type === type ? { ...highlight, enabled: !highlight.enabled } : highlight);
        setHighlightTypes(updatedhighlightTypes);

        const enabledTypes = updatedhighlightTypes
            .filter(highlight => highlight.enabled)
            .map(highlight => highlight.type);

        setHighlightData(enabledTypes);
    };

    const renderHighlightTypeItem = (highlight: { type: string, label: string, enabled: boolean, icon: string, description: string}) => {
        return (
            <List.Item
                className={`highlight-type-item ${highlight.enabled ? 'enabled' : 'disabled'}`}
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-start',
                    width: '100%',
                    paddingLeft: '4px',
                }}
            >
                <div
                    className="highlight-header"
                    style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}
                >
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                        <div className="highlight-icon" style={{ marginRight: '8px', display: 'flex', alignItems: 'center' }}>
                            {highlight.icon === 'error'
                                ? <XCircle size={13} color="var(--color-error)" />
                                : <RefreshCw size={13} color="var(--color-warning)" />}
                        </div>
                        <div className="highlight-label" style={{ fontSize: 11, color: 'var(--text-primary)' }}>
                            {highlight.label}
                        </div>
                    </div>
                    <div className="highlight-toggle" style={{ marginRight: '10px' }}>
                        <Switch
                            size="small"
                            checked={highlight.enabled}
                            onChange={() => handleToggleHighlightType(highlight.type)}
                        />
                    </div>
                </div>
            </List.Item>
        );
    };

    return (
        <div
            className="highlight-detection-container"
        >
            <List
                size="small"
                bordered={false}
                dataSource={highlightTypes}
                renderItem={renderHighlightTypeItem}
                locale={{ emptyText: 'No highlight types configured' }}
            />
        </div>
    );
}