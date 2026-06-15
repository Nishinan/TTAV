import React, { useEffect, useRef, useState } from 'react';
import { message, Tabs } from 'antd';
import { MainBlock } from '../component/main-block';
import { FunctionPanel } from '../component/function-panel';
import { TrainingEventPanel } from '../component/training-event-panel';
import InfluenceAnalysisPanel from '../component/influence-panel';
import { TokenPanel } from '../component/token-panel';
import { useDefaultStore,useGlobalStore } from '../state/state.unified';
import * as BackendAPI from '../communication/backend';

import "../index.css";
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { REFINE_DEFAULTS } from '../config/refine';

const LOG_PREFIX = '[TTVisualizer]';

function logWithTimestamp(message: string): void {
    console.log(`${LOG_PREFIX}[${new Date().toISOString()}] ${message}`);
}

interface EIFJumpPayload {
    source?: 'eif';
    sampleId?: string;
    contentPath?: string;
    visMethod?: string;
    visId?: string;
    dataType?: 'Text' | 'Image';
    taskType?: string;
    selectedIndices?: number[];
    targetIndex?: number;
    selectedSourceIndex?: number;
}

function parseEIFJumpPayloadFromLocation(): EIFJumpPayload | null {
    if (typeof window === 'undefined') return null;
    const params = new URLSearchParams(window.location.search);
    const rawPayload = params.get('eif_jump');
    if (!rawPayload) return null;

    params.delete('eif_jump');
    const nextSearch = params.toString();
    const nextUrl = `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ''}${window.location.hash}`;
    window.history.replaceState({}, '', nextUrl);

    try {
        return JSON.parse(rawPayload) as EIFJumpPayload;
    } catch {
        return null;
    }
}

function normalizeSelectedIndices(value: number[] | undefined): number[] {
    if (!Array.isArray(value)) return [];
    const deduped = new Set<number>();
    value.forEach((idx) => {
        if (Number.isInteger(idx) && idx >= 0) deduped.add(idx);
    });
    return Array.from(deduped).sort((a, b) => a - b);
}

function normalizeEIFSessionInfo(response: any) {
    const info = response?.eifSessionInfo;
    if (response?.eifBundle !== true || !info || typeof info !== 'object') {
        return null;
    }

    return {
        isEifBundle: true,
        sampleId: typeof info.sample_id === 'string' ? info.sample_id : '',
        trainableSessionStatus: typeof response.trainableSessionStatus === 'string'
            ? response.trainableSessionStatus
            : (typeof info.trainable_session_status === 'string' ? info.trainable_session_status : 'registered'),
        refineReady: response.refineReady === true || info.refine_ready === true,
        message: typeof response.message === 'string'
            ? response.message
            : (typeof info.message === 'string' ? info.message : 'Adaptive refine session is still preparing.'),
        updatedAt: typeof info.updated_at === 'number' ? info.updated_at : Date.now(),
    };
}

// // 1. 定义接口，明确告诉 TypeScript 这个组件接受什么属性
// interface FunctionViewPanelsProps {
//     onFocusModeChange: (mode: string) => Promise<void>;
// }
// 修改 Props 接口，增加 onUpdateProjection
interface FunctionViewPanelsProps {
    onUpdateProjection: () => Promise<void>;
    refineReady?: boolean;
    refineStatusMessage?: string | null;
}


  const loadSingleEpoch = async (
        contentPath: string, method: string, visID: string, epochNum: number, taskType: string,
        refineFlag: boolean = false  // when true, load from the _refined projection directory
    ) => {
        // Fire all independent requests in parallel to minimize round-trip latency.
        const isClassification = taskType === 'Classification';

        const [projection, originalNeighbors, projectionNeighbors, predictionResponse, background] =
            await Promise.all([
                BackendAPI.fetchEpochProjection(contentPath, method, visID, epochNum, refineFlag),
                BackendAPI.getOriginalNeighbors(contentPath, epochNum),
                BackendAPI.getProjectionNeighbors(contentPath, method, visID, epochNum, refineFlag),
                isClassification
                    ? BackendAPI.getAttributeResource(contentPath, epochNum, 'prediction')
                    : Promise.resolve({ prediction: [] }),
                isClassification
                    ? BackendAPI.getBackground(contentPath, method, visID, epochNum)
                    : Promise.resolve(''),
            ]);

        const rawProjection = projection.projection || [];
        const data: any = {
            projection: rawProjection,
            originalProjection: rawProjection,  // never overwritten after refine
            originalNeighbors: originalNeighbors.neighbors || [],
            projectionNeighbors: projectionNeighbors.neighbors || [],
            indexList: projectionNeighbors.index_list || [],
        };

        if (isClassification) {
            const prob = predictionResponse.prediction || [];
            data['predProbability'] = prob;
            data['prediction'] = prob.map((p: number[]) => p.indexOf(Math.max(...p)));
            data['background'] = background || '';
        }
        return data;
    };

const DEFAULT_BLEND_DECAY_RATIO = REFINE_DEFAULTS.blendDecayRatio;

function distanceToBBox(x: number, y: number, bbox: { xMin: number; xMax: number; yMin: number; yMax: number }): number {
    const dx = x < bbox.xMin ? bbox.xMin - x : (x > bbox.xMax ? x - bbox.xMax : 0);
    const dy = y < bbox.yMin ? bbox.yMin - y : (y > bbox.yMax ? y - bbox.yMax : 0);
    return Math.sqrt(dx * dx + dy * dy);
}

function buildBlendedProjection(
    baselineProjection: number[][],
    refinedProjection: number[][],
    bbox: { xMin: number; xMax: number; yMin: number; yMax: number } | null,
    focusIndices: number[] = [],
    decayRatio: number = DEFAULT_BLEND_DECAY_RATIO,
): number[][] {
    if (baselineProjection.length !== refinedProjection.length) {
        return refinedProjection;
    }

    const bboxWidth = bbox ? Math.max(Math.abs(bbox.xMax - bbox.xMin), 1e-6) : 1e-6;
    const bboxHeight = bbox ? Math.max(Math.abs(bbox.yMax - bbox.yMin), 1e-6) : 1e-6;
    const bboxDecay = Math.max(Math.sqrt(bboxWidth * bboxWidth + bboxHeight * bboxHeight) * decayRatio, 1e-6);

    const validFocusIndices = focusIndices.filter((idx) => idx >= 0 && idx < baselineProjection.length);
    const focusCoords = validFocusIndices.map((idx) => baselineProjection[idx]);
    let focusDecay = bboxDecay;
    if (focusCoords.length > 1) {
        const center = focusCoords.reduce(
            (acc, point) => [acc[0] + point[0], acc[1] + point[1]],
            [0, 0]
        ).map((v) => v / focusCoords.length) as [number, number];
        const radii = focusCoords.map((point) => Math.hypot(point[0] - center[0], point[1] - center[1]));
        const sorted = [...radii].sort((a, b) => a - b);
        const q75 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.75))] ?? 0;
        focusDecay = Math.max(q75 * 1.5, bboxDecay * 0.5, 1e-6);
    }

    return baselineProjection.map((baselinePoint, idx) => {
        const refinedPoint = refinedProjection[idx] ?? baselinePoint;
        const bboxWeight = bbox
            ? (() => {
                const d = distanceToBBox(baselinePoint[0], baselinePoint[1], bbox);
                return d <= 1e-12 ? 1 : Math.exp(-d / bboxDecay);
            })()
            : 0;
        const focusWeight = validFocusIndices.length > 0
            ? (validFocusIndices.includes(idx)
                ? 1
                : Math.exp(-Math.min(...focusCoords.map((point) => Math.hypot(baselinePoint[0] - point[0], baselinePoint[1] - point[1]))) / focusDecay))
            : 0;
        const weight = Math.max(bboxWeight, focusWeight);
        return [
            baselinePoint[0] * (1 - weight) + refinedPoint[0] * weight,
            baselinePoint[1] * (1 - weight) + refinedPoint[1] * weight,
        ];
    });
}

const initStaticContext = async (contentPath: string, dataType: string) => {
    // 1. 获取训练进程的基础信息
    const processInfo = await BackendAPI.fetchTrainingProcessInfo(contentPath);
    
    // 2. 构造颜色和标签字典 (用于点的着色)
    const colorMap = new Map();
    const labelMap = new Map();
    if (processInfo.color_list) {
        processInfo.color_list.forEach((color: number[], i: number) => {
            colorMap.set(i, [color[0], color[1], color[2]]);
            labelMap.set(i, processInfo.label_text_list[i]);
        });
    }

    // 3. 【关键修复】加载点的固有类别标签 (Inherent Labels)
    const labelsResponse = await BackendAPI.getAttributeResource(contentPath, processInfo.available_epochs[0], 'label');
    const inherentLabelData = labelsResponse.label || [];

    // 4. 【关键修复】加载文本数据和 Token (不能传空！)
    let textData: any[] = [];
    let tokenList: any[] = [];
    if (dataType === 'Text') {
        console.log("[TTAV] Fetching full text data and tokens...");
        const textResponse = await BackendAPI.getText(contentPath);
        textData = textResponse.text_data || [];
        tokenList = textResponse.token_list || [];
    }

    return { 
        processInfo, 
        colorMap, 
        labelMap, 
        inherentLabelData,
        textInfo: { 
            data: textData, 
            tokens: tokenList 
        } 
    };
};

/**
 * Evaluate projection quality after refinement.
 * Reports focus displacement, global drift, neighbor preservation, and trustworthiness.
 */
export const evaluateProjectionQuality = async (
    epochNum: number,
    selectedIndices: number[],
    oldData: any, // snapshot before refine
    newData: any  // data after refine
) => {
    console.log(`\n[Quality Evaluation] Starting evaluation for Epoch ${epochNum}...`);

    const oldProj = oldData.projection;
    const newProj = newData.projection;

    if (!oldProj || !newProj) return;

    // Build reverse mapping: raw dataset index → array position
    // indexList[pos] = rawIdx  =>  rawToPos[rawIdx] = pos
    const indexList: number[] = (newData.indexList && newData.indexList.length > 0)
        ? newData.indexList
        : (oldData.indexList || []);
    const rawToPos = new Map<number, number>();
    indexList.forEach((rawIdx: number, pos: number) => rawToPos.set(rawIdx, pos));
    // Helper: convert a raw dataset index to its array position (identity fallback)
    const toPos = (rawIdx: number) => rawToPos.has(rawIdx) ? rawToPos.get(rawIdx)! : rawIdx;

    // --- Dim 1: Focus Displacement ---
    // Average 2D movement of selected (focus) points — measures refinement effect.
    let focusShift = 0;
    selectedIndices.forEach(rawIdx => {
        const pos = toPos(rawIdx);
        if (!oldProj[pos] || !newProj[pos]) return;
        const d = Math.sqrt(
            Math.pow(newProj[pos][0] - oldProj[pos][0], 2) +
            Math.pow(newProj[pos][1] - oldProj[pos][1], 2)
        );
        focusShift += d;
    });
    const avgFocusShift = focusShift / (selectedIndices.length || 1);

    // --- Dim 2: Global Stability (Drift) ---
    // Average 2D movement of non-selected points — smaller is more stable.
    let globalDrift = 0;
    let nonFocusCount = 0;
    const selectedPosSet = new Set(selectedIndices.map(toPos));

    oldProj.forEach((pos: number[], i: number) => {
        if (!selectedPosSet.has(i)) {
            const d = Math.sqrt(
                Math.pow(newProj[i][0] - pos[0], 2) +
                Math.pow(newProj[i][1] - pos[1], 2)
            );
            globalDrift += d;
            nonFocusCount++;
        }
    });
    const avgGlobalDrift = globalDrift / (nonFocusCount || 1);

    // --- Dim 3: Neighbor Preservation ---
    // How many of the post-refine low-D neighbors are also high-D neighbors?
    // Both originalNeighbors and projectionNeighbors store proj-position indices (0..N-1),
    // because both are built on arrays re-ordered by index.json on the backend.
    // No rawIdx→pos conversion is needed here.
    const highNeighborsNC = oldData.originalNeighbors || [];
    const newLowNeighbors = newData.projectionNeighbors || [];



    let neighborConsistency = 0;
    let ncCount = 0;
    selectedIndices.forEach(rawIdx => {
        const pos = toPos(rawIdx);
        const highList: number[] = highNeighborsNC[pos] || [];
        const lowList:  number[] = newLowNeighbors[pos] || [];
        if (highList.length === 0 || lowList.length === 0) return;

        const k = Math.min(highList.length, lowList.length);
        const highSet = new Set(highList.slice(0, k));
        const intersection = lowList.slice(0, k).filter(p => highSet.has(p));
        neighborConsistency += intersection.length / k;
        ncCount++;
    });
    const avgNeighborConsistency = ncCount > 0 ? neighborConsistency / ncCount : 0;

    // --- Dim 4: Trustworthiness ---
    // T = 1 - (2 / n·k·(2n-3k-1)) × Σ_i Σ_{j∈U_i} (r(i,j) - k)
    // U_i = points in low-D neighborhood but not in high-D neighborhood.
    // Both neighbor lists use the same proj-position index space — no conversion needed.
    const highNeighborsTrust = oldData.originalNeighbors || [];
    const lowNeighborsTrust  = newData.projectionNeighbors || [];

    let trustSum = 0;
    let trustCount = 0;
    let continuitySum = 0;
    let continuityCount = 0;
    const N = newProj.length;

    selectedIndices.forEach(rawIdx => {
        const pos = toPos(rawIdx);
        const highList: number[] = highNeighborsTrust[pos] || [];
        const lowList:  number[] = lowNeighborsTrust[pos]  || [];
        if (highList.length === 0 || lowList.length === 0) return;

        const k = Math.min(highList.length, lowList.length);
        const highSet = new Set(highList.slice(0, k));

        // Penalty: low-D neighbor j not found in high-D top-k
        let penalty = 0;
        lowList.slice(0, k).forEach(j => {
            if (!highSet.has(j)) {
                const rank = highList.indexOf(j);
                const r = rank === -1 ? highList.length + 1 : rank + 1;
                penalty += (r - k);
            }
        });

        const norm = k * (2 * N - 3 * k - 1) / 2;
        trustSum += norm > 0 ? 1 - penalty / norm : 1;
        trustCount++;

        let continuityPenalty = 0;
        highList.slice(0, k).forEach(j => {
            if (!lowList.slice(0, k).includes(j)) {
                const rank = lowList.indexOf(j);
                const r = rank === -1 ? lowList.length + 1 : rank + 1;
                continuityPenalty += (r - k);
            }
        });
        continuitySum += norm > 0 ? 1 - continuityPenalty / norm : 1;
        continuityCount++;
    });
    const avgTrustworthiness = trustCount > 0 ? trustSum / trustCount : 1;
    const avgContinuity = continuityCount > 0 ? continuitySum / continuityCount : 1;

    console.log("-----------------------------------------");
    console.log(`> Focus Displacement: ${avgFocusShift.toFixed(4)}`);
    console.log(`> Global Drift: ${avgGlobalDrift.toFixed(4)}`);
    console.log(`> Neighbor Preservation: ${(avgNeighborConsistency * 100).toFixed(2)}%`);
    console.log(`> Trustworthiness: ${(avgTrustworthiness * 100).toFixed(2)}%`);
    console.log(`> Continuity: ${(avgContinuity * 100).toFixed(2)}%`);
    console.log("-----------------------------------------");

    return {
        avgFocusShift,
        avgGlobalDrift,
        avgNeighborConsistency,
        avgTrustworthiness,
        avgContinuity,
    };
};


/**
 * Compute displacement statistics: focus shift vs global drift.
 */
export const calculateDisplacementStats = (
    oldProj: number[][],
    newProj: number[][],
    selectedIndices: number[],
    indexList: number[] = []
) => {
    if (!oldProj || !newProj || oldProj.length !== newProj.length) {
        console.error("Invalid projection data for displacement stats.");
        return null;
    }

    // Build raw → pos mapping if indexList provided
    const rawToPos = new Map<number, number>();
    indexList.forEach((rawIdx, pos) => rawToPos.set(rawIdx, pos));
    const toPos = (rawIdx: number) => rawToPos.has(rawIdx) ? rawToPos.get(rawIdx)! : rawIdx;

    let focusShiftTotal = 0;
    let globalDriftTotal = 0;
    const selectedPosSet = new Set(selectedIndices.map(toPos));
    const numTotal = oldProj.length;
    const numFocus = selectedIndices.length;
    const numNonFocus = numTotal - numFocus;

    for (let i = 0; i < numTotal; i++) {
        const dx = newProj[i][0] - oldProj[i][0];
        const dy = newProj[i][1] - oldProj[i][1];
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (selectedPosSet.has(i)) {
            focusShiftTotal += distance;
        } else {
            globalDriftTotal += distance;
        }
    }

    const avgFocusShift = numFocus > 0 ? focusShiftTotal / numFocus : 0;
    const avgGlobalDrift = numNonFocus > 0 ? globalDriftTotal / numNonFocus : 0;

    console.log(`\n[Displacement Stats]`);
    console.log(`- Avg Focus Shift: ${avgFocusShift.toFixed(5)}`);
    console.log(`- Avg Global Drift: ${avgGlobalDrift.toFixed(5)}`);
    const snr = avgGlobalDrift > 0 ? avgFocusShift / avgGlobalDrift : Infinity;
    console.log(`- Stability Ratio: ${snr.toFixed(2)}x`);

    return {
        avgFocusShift,
        avgGlobalDrift,
        stabilityRatio: snr
    };
};

// 1. 在组件外部定义 refreshEpochData
// 无需使用 Hook，直接引用 store 实例

const refreshEpochData = async (
    epochNum: number,
    params: { contentPath: string; vis_method: string; visID: string; taskType: string },
    updateGlobal: boolean = true,
    refineFlag: boolean = false  // propagated to loadSingleEpoch for post-refine neighbor accuracy
) => {
    const epochData = await loadSingleEpoch(
        params.contentPath,
        params.vis_method,
        params.visID,
        epochNum,
        params.taskType,
        refineFlag
    );

    const curP = epochData.projection;
    const minX = Math.min(...curP.map((p: any) => p[0])), maxX = Math.max(...curP.map((p: any) => p[0]));
    const minY = Math.min(...curP.map((p: any) => p[1])), maxY = Math.max(...curP.map((p: any) => p[1]));

    // 直接从 store 实例获取当前快照，无需 Hook
    const state = useGlobalStore.getState(); 
    const currentBounds = state.globalBounds;

    const newBounds = {
        minX: Math.min(currentBounds?.minX ?? Infinity, minX),
        maxX: Math.max(currentBounds?.maxX ?? -Infinity, maxX),
        minY: Math.min(currentBounds?.minY ?? Infinity, minY),
        maxY: Math.max(currentBounds?.maxY ?? -Infinity, maxY),
    };

    if (updateGlobal) {
        // 直接使用 getState 里的 setValue 触发更新
        state.setValue('allEpochData', {
            ...state.allEpochData,
            [epochNum]: epochData
        });
        state.setValue('globalBounds', newBounds);
    }

    return epochData;
};

// MessageHandler component for handling extension communication and backend requests
function MessageHandler() {
    // State from unified store
    const {
        contentPath, visID, vis_method, dataType, taskType, eifSessionInfo,
        setContentPath, setAvailableEpochs, setDataType, setTaskType,
        setTextData, setTokenList, setInherentLabelData,
        setColorDict, setLabelDict, setProgress, setValue, setSelectedIndices, setHoveredIndex,
    } = useDefaultStore([
        'contentPath', 'visID', 'vis_method', 'dataType', 'taskType', 'eifSessionInfo',
        'setContentPath', 'setAvailableEpochs', 'setDataType', 'setTaskType',
        'setTextData', 'setTokenList', 'setInherentLabelData',
        'setColorDict', 'setLabelDict', 'setProgress', 'setValue', 'setSelectedIndices', 'setHoveredIndex'
    ]);

    // Start visualizing process
    const handleStartVisualizing = async (
        contentPath: string,
        visualizationMethod: string,
        visualizationID: string,
        dataType: string,
        taskType: string,
        visConfig: any
    ) => {
        try {
            let startTime = Date.now();
            await BackendAPI.triggerStartVisualizing(contentPath, visualizationMethod, visualizationID, dataType, taskType, visConfig);
            logWithTimestamp(`Visualization process started in backend. timeCost=${Date.now() - startTime}ms`);
        } catch (error:any) {
            console.error('Error starting visualization process:', error);
            message.error('Failed to start visualization process');
            // 这里捕获 Server 返回的 409 报错
            const serverMsg = error.response?.data?.message || "Unknown error";
            message.error(`Start Failed: ${serverMsg}`);
            }
    }

   const handleSyncSession = async (
    contentPath: string, 
    visualizationMethod: string, 
    visualizationID: string, 
    dataType: string, 
    taskType: string, 
    visConfig: any
) => {
    try {
        console.log("[TTAV] Manually syncing session with ID:", visualizationID);
        // 显示加载状态（由于不传大数据，通常很快）
        message.loading({ content: 'Syncing backend...', key: 'sync_task' });

        const fullSyncConfig = {
            content_path: contentPath,
            vis_method: visualizationMethod,
            visualizationID: visualizationID,
            data_type: dataType,
            task_type: taskType,
            vis_config: visConfig || { gpu_id: -1 }
        };

        const response = await BackendAPI.syncSession(fullSyncConfig);

        if (response.status === "success") {
            message.success({ content: 'Backend Session Resumed!', key: 'sync_task' });
            const normalizedEIFSessionInfo = normalizeEIFSessionInfo(response);
            setValue('eifSessionInfo', normalizedEIFSessionInfo);
            // 更新当前路径等基础状态，确保后续 Update 正常
            setContentPath(contentPath);
            setDataType(dataType as 'Text' | 'Image');
            setTaskType(taskType);
            setValue('visID', visualizationID);
            setValue('vis_method', visualizationMethod);
        } else {
            throw new Error(response.message);
        }
    } catch (error: any) {
        console.error('Error syncing session:', error);
        message.error({ content: `Sync failed: ${error.message}`, key: 'sync_task' });
    }
};
    const handleLoadVisualization = async (
        contentPath: string, 
        visualizationMethod: string, 
        visualizationID: string, 
        dataType: string, 
        taskType: string, 
        visConfig: any
    ) => {
        try {
            logWithTimestamp(`[TTAV] Start loading visualization: ${visualizationID}`);

            // Clear stale epoch data (e.g. from a previous refine) before loading fresh data.
            // Without this, the old refined projectionNeighbors remain in the store while epochs
            // load one-by-one, and the user may interact with stale data mid-load.
            useGlobalStore.getState().setValue('allEpochData', {});

            const staticCtx = await initStaticContext(contentPath, dataType);

            // 同步所有静态上下文
            setColorDict(staticCtx.colorMap);
            setLabelDict(staticCtx.labelMap);
            setInherentLabelData(staticCtx.inherentLabelData); // 修复颜色变色
            setTextData(staticCtx.textInfo.data);              // 修复文本丢失
            setTokenList(staticCtx.textInfo.tokens);            // 修复 Token 丢失
            
            const epochs = staticCtx.processInfo.available_epochs || [];
            setAvailableEpochs(epochs);
           
            for (const epochNum of epochs) {
                await refreshEpochData(epochNum, { contentPath, vis_method: visualizationMethod, visID:visualizationID, taskType },true);
                
                // 更新进度条
                setProgress(((epochs.indexOf(epochNum) + 1) / epochs.length) * 100);
            }

            // 4. 同步后端 Session
            const syncResponse = await BackendAPI.syncSession({
                content_path: contentPath, vis_method: visualizationMethod, vis_id: visualizationID,
                data_type: dataType, task_type: taskType, vis_config: visConfig
            });
            setValue('eifSessionInfo', normalizeEIFSessionInfo(syncResponse));

            // 5. 更新 store，确保后续 handleUpdate 能拿到正确的 vis_method / visID
            setContentPath(contentPath);
            setDataType(dataType as 'Text' | 'Image');
            setTaskType(taskType);
            setValue('vis_method', visualizationMethod);
            setValue('visID', visualizationID);

            message.success('Visualization loaded successfully!');
            return true;
        } catch (error) {
            console.error('Error:', error);
            message.error('Failed to load visualization');
            return false;
        }
    };

    const applyEIFHighlightUpdate = (payload: EIFJumpPayload) => {
        const selected = normalizeSelectedIndices(payload.selectedIndices);
        setSelectedIndices(selected);
        setHoveredIndex(typeof payload.targetIndex === 'number' ? payload.targetIndex : undefined);
    };

    // 增加一个 Ref 锁，防止同一 ID 的任务被重复触发
    const processingMessageIds = useRef(new Set<string>());

    const handleMessage = async (event: MessageEvent) => {
        const { command, data } = event.data;
        console.log('Received message from extension:', event);

        if (command === 'eifHighlightUpdate') {
            const currentContentPath = useGlobalStore.getState().contentPath;
            if (!data?.contentPath || (currentContentPath && data.contentPath !== currentContentPath)) {
                return;
            }
            applyEIFHighlightUpdate(data as EIFJumpPayload);
            return;
        }

        // 如果插件没传 id，可以用 command + contentPath 组合成简单锁
        const lockKey = `${command}-${data?.contentPath}`;
        if (processingMessageIds.current.has(lockKey)) return;

        const vis_id = data.visualizationID ? data.visualizationID : 0;

        try {
            processingMessageIds.current.add(lockKey);
            switch (command) {
                case 'startVisualizing':
                    await handleStartVisualizing(data.contentPath, data.visualizationMethod, vis_id, data.dataType, data.taskType, data.visConfig);
                    break;
                case 'loadVisualization':
                    await handleLoadVisualization(data.contentPath, data.visualizationMethod, vis_id, data.dataType, data.taskType, data.visConfig);
                    break;
                case 'syncSession':
                    await handleSyncSession(data.contentPath, data.visualizationMethod, vis_id, data.dataType, data.taskType, data.visConfig);
                    break;
                default:
                    console.log('Unknown message command:', command);
            }
        }
         finally {
            // 执行完后移除锁
            processingMessageIds.current.delete(lockKey);
        }
    };

    useEffect(() => {
        window.addEventListener('message', handleMessage);

        return () => window.removeEventListener('message', handleMessage);
    }, []);

    const eifReadyReloadRef = useRef(new Set<string>());

    useEffect(() => {
        if (!eifSessionInfo?.isEifBundle || eifSessionInfo.refineReady || !contentPath || !vis_method || !visID) {
            return;
        }

        const reloadKey = `${contentPath}::${vis_method}::${visID}`;
        let active = true;
        let intervalId: number | null = null;

        const pollStatus = async () => {
            try {
                const response = await BackendAPI.getEIFBundleStatus(contentPath, vis_method, visID);
                if (!active || response?.status !== 'success') return;

                const normalizedEIFSessionInfo = normalizeEIFSessionInfo(response);
                setValue('eifSessionInfo', normalizedEIFSessionInfo);

                if (normalizedEIFSessionInfo?.refineReady && !eifReadyReloadRef.current.has(reloadKey)) {
                    eifReadyReloadRef.current.add(reloadKey);
                    message.loading({
                        content: 'Adaptive refine session is ready. Refreshing projection...',
                        key: 'eif_refine_ready_reload',
                        duration: 0,
                    });
                    const reloaded = await handleLoadVisualization(
                        contentPath,
                        vis_method,
                        visID,
                        dataType,
                        taskType,
                        { gpu_id: -1 }
                    );
                    if (reloaded) {
                        message.success({
                            content: 'Adaptive refine session is ready. Projection refreshed.',
                            key: 'eif_refine_ready_reload',
                        });
                    } else {
                        message.warning({
                            content: 'Adaptive refine session is ready, but the automatic refresh failed.',
                            key: 'eif_refine_ready_reload',
                        });
                    }
                }
            } catch (error) {
                console.error('[TTAV] Failed to poll EIF bundle status:', error);
            }
        };

        void pollStatus();
        intervalId = window.setInterval(() => {
            void pollStatus();
        }, 3000);

        return () => {
            active = false;
            if (intervalId !== null) {
                window.clearInterval(intervalId);
            }
        };
    }, [contentPath, dataType, eifSessionInfo, setValue, taskType, visID, vis_method]);

    useEffect(() => {
        const payload = parseEIFJumpPayloadFromLocation();
        if (!payload?.contentPath) return;

        const contentPath = payload.contentPath;
        const visMethod = payload.visMethod || 'UMAP';
        const visId = payload.visId || '1';
        const dataType = payload.dataType || 'Text';
        const taskType = payload.taskType || 'Alignment';
        const selected = normalizeSelectedIndices(payload.selectedIndices);

        void (async () => {
            const loaded = await handleLoadVisualization(
                contentPath,
                visMethod,
                visId,
                dataType,
                taskType,
                { gpu_id: -1 }
            );
            if (!loaded) return;
            applyEIFHighlightUpdate(payload);
            const sampleLabel = payload.sampleId ? ` ${payload.sampleId}` : '';
            message.success(`EIF jump loaded${sampleLabel}. ${selected.length} token(s) selected.`);
        })();
    }, []);

    return <></>;
}

// Normalize raw backend metrics response into a consistent shape (or null if absent).
function normalizeStructuralMetrics(raw: any): {
    neighbor_preservation: number | null;
    mean_rank_hd: number | null;
    trustworthiness: number | null;
    continuity: number | null;
} | null {
    if (!raw) return null;
    const np = raw.neighbor_preservation ?? null;
    const mrh = raw.mean_rank_hd ?? null;
    const t = raw.trustworthiness ?? null;
    const c = raw.continuity ?? null;
    if (np === null && mrh === null && t === null && c === null) return null;
    return { neighbor_preservation: np, mean_rank_hd: mrh, trustworthiness: t, continuity: c };
}

// Evaluate projection quality and write refineMetrics to the global store.
async function updateDisplayedRefineMetrics(
    focusIndices: number[],
    oldEpochData: any,
    newEpochData: any,
    backendMetrics: ReturnType<typeof normalizeStructuralMetrics>,
) {
    const frontendResult = await evaluateProjectionQuality(0, focusIndices, oldEpochData, newEpochData);
    const state = useGlobalStore.getState();
    // Backend values are already percentages (0-100). Panel displays as `v * 100 %`,
    // so divide by 100 to store as fraction (0-1).
    // Frontend fallback values (avgNeighborConsistency etc.) are already fractions (0-1).
    state.setValue('refineMetrics', {
        focusDisplacement: frontendResult?.avgFocusShift ?? 0,
        globalDrift: frontendResult?.avgGlobalDrift ?? 0,
        neighborPreservation: backendMetrics?.neighbor_preservation != null
            ? backendMetrics.neighbor_preservation / 100
            : (frontendResult?.avgNeighborConsistency ?? 0),
        meanRankHD: backendMetrics?.mean_rank_hd ?? 0,
        trustworthiness: backendMetrics?.trustworthiness != null
            ? backendMetrics.trustworthiness / 100
            : (frontendResult?.avgTrustworthiness ?? 0),
        continuity: backendMetrics?.continuity != null
            ? backendMetrics.continuity / 100
            : (frontendResult?.avgContinuity ?? 0),
    });
}

export function AppCombinedView() {
    // [TTAV] Deconstruct required state and the generic 'setValue' from the store
    // Note: 'allEpochData' must be included here to be recognized in the function below
    const {
        contentPath,
        selectedIndices,
        visID: currentVisID,
        epoch,
        epoch: targetEpoch,
        vis_method,
        taskType,
        setValue,
        focusMode,
        currentViewportBBox,
        setHoveredIndex,
        eifSessionInfo,
        secondaryIndices,
    } = useDefaultStore([
        'contentPath',
        'selectedIndices',
        'visID',
        'epoch',
        'vis_method',
        'taskType',
        'setValue',
        'focusMode',
        'currentViewportBBox',
        'setHoveredIndex',
        'eifSessionInfo',
        'secondaryIndices',
    ]);
// 用于 Canvas 实时绘制的坐标（这是真正传给 Canvas 组件的数据）
    const [currentDrawingCoords, setCurrentDrawingCoords] = useState<number[][] | null>(null);
    const animationRef = useRef<number>();

    // 平滑平移函数
    const animateTransition = (startCoords: number[][], endCoords: number[][]) => {
        const duration = 800; // 动画持续 800ms
        const startTime = performance.now();

        const step = (currentTime: number) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // 缓动函数 (EaseInOutQuad)
            const ease = progress < 0.5
                ? 2 * progress * progress
                : 1 - Math.pow(-2 * progress + 2, 2) / 2;

            // 计算每一帧的插值坐标
            const interpolated = startCoords.map((start, i) => {
                const end = endCoords[i];
                return [
                    start[0] + (end[0] - start[0]) * ease,
                    start[1] + (end[1] - start[1]) * ease
                ];
            });

            // 更新绘制用的 State
            setCurrentDrawingCoords(interpolated);

            if (progress < 1) {
                animationRef.current = requestAnimationFrame(step);
            } else {
                // 动画结束，正式同步到 Store
                setValue('refinedProjection', endCoords);
            }
        };

        if (animationRef.current) cancelAnimationFrame(animationRef.current);
        animationRef.current = requestAnimationFrame(step);
        };

         

    const isRefining = useRef(false);
    const [, setActiveRefineSessionId] = useState<string | null>(null);
    const REFINE_MSG_KEY = 'ttav_refine_loading';

    const handleUpdate = async () => {
        if (eifSessionInfo?.isEifBundle && !eifSessionInfo.refineReady) {
            message.info(eifSessionInfo.message || 'Adaptive refine session is still preparing.');
            return;
        }
        if (!selectedIndices || selectedIndices.length === 0) {
            message.warning("Please select points on the canvas first.");
            return;
        }
        if (isRefining.current) {
            message.warning("Refinement already in progress, please wait.");
            return;
        }
        isRefining.current = true;
        message.loading({ content: 'Refining layout...', key: REFINE_MSG_KEY, duration: 0 });

        try {
            const oldEpochData = useGlobalStore.getState().allEpochData[epoch];
            const startResponse = await BackendAPI.startRefineSession(
                contentPath,
                selectedIndices,
                focusMode,
                epoch,
                currentViewportBBox,
                secondaryIndices && secondaryIndices.length > 0 ? secondaryIndices : undefined,
            );

            if (startResponse && (startResponse as any).status === "success") {
                const sessionId = (startResponse as any).session_id as string;
                let focusIndices = Array.isArray((startResponse as any).focus_indices)
                    ? (startResponse as any).focus_indices as number[]
                    : selectedIndices;
                let bboxIndices = Array.isArray((startResponse as any).bbox_indices)
                    ? (startResponse as any).bbox_indices as number[]
                    : [];
                let lastProjectionVersion = -1;
                let finalProgress: any = null;
                let latestSampledMetrics: any = null;
                let resolvedBehavior: any = (startResponse as any).resolved_refine_behavior ?? null;
                setActiveRefineSessionId(sessionId);

                // blendBaseline is the pre-refine projection used as the LD anchor for
                // neighbor-set computation during animation (avoids co-location collapse).
                const blendBaseline = oldEpochData.projection;

                const applyIntermediateProjection = (refinedProjection: number[][], latestFocusIndices: number[]) => {
                    const blendedProjection = buildBlendedProjection(
                        blendBaseline,
                        refinedProjection,
                        currentViewportBBox,
                        latestFocusIndices,
                    );
                    const state = useGlobalStore.getState();
                    const currentEpochData = state.allEpochData[targetEpoch] || oldEpochData;
                    state.setValue('focusIndices', latestFocusIndices);
                    state.setValue('allEpochData', {
                        ...state.allEpochData,
                        [targetEpoch]: {
                            ...currentEpochData,
                            projection: blendedProjection,
                            originalProjection: blendBaseline,
                        },
                    });
                    return {
                        ...currentEpochData,
                        projection: blendedProjection,
                        originalProjection: blendBaseline,
                    };
                };

                const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

                while (true) {
                    const progress = await BackendAPI.getRefineSessionProgress(sessionId, lastProjectionVersion);
                    finalProgress = progress;

                    if (Array.isArray((progress as any).focus_indices) && (progress as any).focus_indices.length > 0) {
                        focusIndices = (progress as any).focus_indices as number[];
                    }
                    if (Array.isArray((progress as any).bbox_indices)) {
                        bboxIndices = (progress as any).bbox_indices as number[];
                    }
                    if ((progress as any).resolved_refine_behavior) {
                        resolvedBehavior = (progress as any).resolved_refine_behavior;
                    }
                    const progressMetrics = normalizeStructuralMetrics(
                        (progress as any).sampled_metrics
                        ?? ((progress as any).result ? {
                            neighbor_preservation: (progress as any).result.neighbor_preservation,
                            mean_rank_hd: (progress as any).result.mean_rank_hd,
                            trustworthiness: (progress as any).result.trustworthiness,
                            continuity: (progress as any).result.continuity,
                        } : null)
                    );
                    if (progressMetrics) {
                        latestSampledMetrics = progressMetrics;
                    }

                    if (Array.isArray((progress as any).projection)) {
                        const liveEpochData = applyIntermediateProjection((progress as any).projection as number[][], focusIndices);
                        lastProjectionVersion = Number((progress as any).projection_version ?? lastProjectionVersion);
                        const stepsCompleted = Number((progress as any).steps_completed ?? 0);
                        const maxSteps = Number(
                            resolvedBehavior?.stopping?.max_steps
                            ?? (progress as any)?.resolved_refine_behavior?.stopping?.max_steps
                            ?? 0
                        );
                        await updateDisplayedRefineMetrics(focusIndices, oldEpochData, liveEpochData, latestSampledMetrics);
                        message.loading({
                            content: (progress as any).stop_requested
                                ? `Stopping refine... ${stepsCompleted}${maxSteps > 0 ? ` / ${maxSteps}` : ''} steps`
                                : (stepsCompleted > 0 ? `Refining layout... ${stepsCompleted}${maxSteps > 0 ? ` / ${maxSteps}` : ''} steps` : 'Refining layout...'),
                            key: REFINE_MSG_KEY,
                            duration: 0,
                        });
                    } else if (latestSampledMetrics) {
                        const liveEpochData = useGlobalStore.getState().allEpochData[targetEpoch];
                        if (liveEpochData) {
                            await updateDisplayedRefineMetrics(focusIndices, oldEpochData, liveEpochData, latestSampledMetrics);
                        }
                    }

                    if ((progress as any).status === "completed") {
                        break;
                    }
                    if ((progress as any).status === "failed" || (progress as any).status === "error") {
                        throw new Error((progress as any).error || 'Refine session failed.');
                    }

                    await sleep(REFINE_DEFAULTS.progressPollMs);
                }

                console.log(`[TTAV] Refine success. Fetching updated projection + low-D neighbors...`);
                setActiveRefineSessionId(null);

                const projResp = await BackendAPI.fetchEpochProjection(contentPath, vis_method, currentVisID, targetEpoch, true);
                const refinedProjection = projResp.projection || oldEpochData.projection;
                const projNeighResp = await BackendAPI.getProjectionNeighbors(
                    contentPath,
                    vis_method,
                    currentVisID,
                    targetEpoch,
                    true,
                    currentViewportBBox,
                    focusIndices,
                    DEFAULT_BLEND_DECAY_RATIO,
                );
                const blendedProjection = buildBlendedProjection(
                    blendBaseline,
                    refinedProjection,
                    currentViewportBBox,
                    focusIndices,
                );
                const newEpochData = {
                    ...oldEpochData,
                    projection: blendedProjection,
                    originalProjection: blendBaseline,
                    projectionNeighbors: projNeighResp.neighbors || oldEpochData.projectionNeighbors,
                    indexList: projNeighResp.index_list || oldEpochData.indexList,
                };

                const state = useGlobalStore.getState();
                state.setValue('focusIndices', focusIndices);
                state.setValue('allEpochData', { ...state.allEpochData, [targetEpoch]: newEpochData });
                if (selectedIndices.length > 0) {
                    setHoveredIndex(selectedIndices[0]);
                }

                const backendMetrics = normalizeStructuralMetrics((finalProgress as any)?.result);
                await updateDisplayedRefineMetrics(focusIndices, oldEpochData, newEpochData, backendMetrics);
                calculateDisplacementStats(oldEpochData.projection, newEpochData.projection, focusIndices, newEpochData.indexList || []);

                message.success({ content: `Epoch ${targetEpoch} refined! Refined projection updated!`, key: REFINE_MSG_KEY });
            } else {
                message.error({ content: 'Refinement returned unexpected status.', key: REFINE_MSG_KEY });
            }
        } catch (error) {
            console.error("Update failed:", error);
            message.error({ content: 'Failed to update projection.', key: REFINE_MSG_KEY });
        } finally {
            isRefining.current = false;
        }
    };
    // 1. 监听全局选点，确保 selectedIndices 响应
    useEffect(() => {
    // 只要这个打印了，说明选点通了
    console.log("Global Selection confirmed:", selectedIndices);
}, [selectedIndices]);

    return (
        <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
            <PanelGroup direction="vertical" style={{ flex: 1, display: "flex" }} autoSaveId="plot-view-root">
                <Panel defaultSize={76} minSize={40}>
                    <PanelGroup direction="horizontal" style={{ height: "100%", display: "flex" }} autoSaveId="plot-view-layout">
                        <Panel defaultSize={70} minSize={20}>
                            <div style={{ display: "flex", width: "100%", height: "100%" }}>
                                <MainBlock />
                            </div>
                        </Panel>
                        <PanelResizeHandle className="subtle-resize-handle" hitAreaMargins={{ coarse: 12, fine: 6 }} />
                        <Panel defaultSize={30} minSize={8} maxSize={60} collapsible collapsedSize={0}>
                           <div style={{ width: '100%', height: '100%', borderLeft: '1px solid #ccc' }}>
                
                <FunctionViewPanels
                    onUpdateProjection={handleUpdate}
                    refineReady={!(eifSessionInfo?.isEifBundle) || eifSessionInfo.refineReady}
                    refineStatusMessage={eifSessionInfo?.isEifBundle && !eifSessionInfo.refineReady ? eifSessionInfo.message : null}
                />
            </div>
                        </Panel>
                    </PanelGroup>
                </Panel>
                <PanelResizeHandle className="subtle-resize-handle-horizontal" />
                <Panel defaultSize={24} minSize={8} maxSize={50} collapsible collapsedSize={0}>
                    <div style={{ width: '100%', height: '100%', borderTop: '1px solid #ccc' }}>
                        <BottomDock />
                    </div>
                </Panel>
            </PanelGroup>
            <MessageHandler />
        </div>
    );
}

// 2. 修改组件定义，使其接收 Props
export function FunctionViewPanels({ onUpdateProjection, refineReady = true, refineStatusMessage = null }: FunctionViewPanelsProps) {
    const [activeKey, setActiveKey] = useState<'FunctionPanel' | 'TrainingEventPanel'>('FunctionPanel');

    const items = [
        { key: 'FunctionPanel', label: <span style={{ fontSize: 12 }}>Functions</span> },
        { key: 'TrainingEventPanel', label: <span style={{ fontSize: 12 }}>Training Events</span> },
    ];

    return (
        <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Tabs
                className="function-tabs"
                activeKey={activeKey}
                onChange={(key) => setActiveKey(key as typeof activeKey)}
                size="small"
                tabBarStyle={{ marginBottom: 0 }}
                tabBarGutter={0}
                items={items}
            />
            <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
                {activeKey === 'FunctionPanel' && (
                    <FunctionPanel
                        onUpdateProjection={onUpdateProjection}
                        refineReady={refineReady}
                        refineStatusMessage={refineStatusMessage}
                    />
                )}
                {activeKey === 'TrainingEventPanel' && <TrainingEventPanel />}
            </div>
        </div>
    );
}

function BottomDock() {
    const [activeKey, setActiveKey] = useState<'Influence' | 'Tokens'>('Influence');
    const items = [
        { key: 'Influence', label: <span style={{ fontSize: 12 }}>Influence</span>, children: <InfluenceAnalysisPanel /> },
        { key: 'Tokens', label: <span style={{ fontSize: 12 }}>Tokens</span>, children: <TokenPanel /> },
    ];

    return (
        <Tabs
            className="bottom-dock-tabs"
            tabPosition="right"
            size="small"
            tabBarGutter={0}
            tabBarStyle={{ marginLeft: 0 }}
            style={{ height: '100%' }}
            items={items}
            activeKey={activeKey}
            onChange={(key) => setActiveKey(key as typeof activeKey)}
        />
    );
}

window.vscode?.postMessage({ state: 'load' }, '*');
