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

const LOG_PREFIX = '[TTVisualizer]';

function logWithTimestamp(message: string): void {
    console.log(`${LOG_PREFIX}[${new Date().toISOString()}] ${message}`);
}

// // 1. 定义接口，明确告诉 TypeScript 这个组件接受什么属性
// interface FunctionViewPanelsProps {
//     onFocusModeChange: (mode: string) => Promise<void>;
// }
// 修改 Props 接口，增加 onUpdateProjection
interface FunctionViewPanelsProps {
    onUpdateProjection: () => Promise<void>;
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

        const data: any = {
            projection: projection.projection || [],
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
    });
    const avgTrustworthiness = trustCount > 0 ? trustSum / trustCount : 1;

    console.log("-----------------------------------------");
    console.log(`> Focus Displacement: ${avgFocusShift.toFixed(4)}`);
    console.log(`> Global Drift: ${avgGlobalDrift.toFixed(4)}`);
    console.log(`> Neighbor Preservation: ${(avgNeighborConsistency * 100).toFixed(2)}%`);
    console.log(`> Trustworthiness: ${(avgTrustworthiness * 100).toFixed(2)}%`);
    console.log("-----------------------------------------");

    return {
        avgFocusShift,
        avgGlobalDrift,
        avgNeighborConsistency,
        avgTrustworthiness,
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
        setContentPath, setAvailableEpochs, setDataType, setTaskType,
        setTextData, setTokenList, setInherentLabelData,
        setColorDict, setLabelDict, setProgress, setValue,
    } = useDefaultStore([
        'setContentPath', 'setAvailableEpochs', 'setDataType', 'setTaskType',
        'setTextData', 'setTokenList', 'setInherentLabelData',
        'setColorDict', 'setLabelDict', 'setProgress', 'setValue'
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
            // 更新当前路径等基础状态，确保后续 Update 正常
            setContentPath(contentPath);
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
            await BackendAPI.syncSession({
                content_path: contentPath, vis_method: visualizationMethod, vis_id: visualizationID,
                data_type: dataType, task_type: taskType, vis_config: visConfig
            });

            // 5. 更新 store，确保后续 handleUpdate 能拿到正确的 vis_method / visID
            setContentPath(contentPath);
            setValue('vis_method', visualizationMethod);
            setValue('visID', visualizationID);

            message.success('Visualization loaded successfully!');
        } catch (error) {
            console.error('Error:', error);
            message.error('Failed to load visualization');
        }
    };

    // 增加一个 Ref 锁，防止同一 ID 的任务被重复触发
    const processingMessageIds = useRef(new Set<string>());

    const handleMessage = async (event: MessageEvent) => {
        const { command, data } = event.data;
        console.log('Received message from extension:', event);

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

    return <></>;
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
    } = useDefaultStore([
        'contentPath',
        'selectedIndices',
        'visID',
        'epoch',
        'vis_method',
        'taskType',
        'setValue',
        'focusMode',
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
    const REFINE_MSG_KEY = 'ttav_refine_loading';

    const handleUpdate = async () => {
        if (!selectedIndices || selectedIndices.length === 0) {
            message.warning("Please select points on the canvas first.");
            return;
        }
        // Prevent concurrent refine calls — each would create its own loading toast
        if (isRefining.current) {
            message.warning("Refinement already in progress, please wait.");
            return;
        }
        isRefining.current = true;
        // Use a stable key so any stale toast from a prior crash is destroyed first
        message.loading({ content: 'Refining layout...', key: REFINE_MSG_KEY, duration: 0 });

        try {
            const oldEpochData = useGlobalStore.getState().allEpochData[epoch];
            const response = await BackendAPI.updateFocusContext(contentPath, selectedIndices, focusMode, epoch);

            if (response && response.status === "success") {
                console.log(`[TTAV] Refine success. Fetching updated projection + low-D neighbors...`);

                // After refine, only projection coords and low-D neighbors change.
                // Re-use originalNeighbors/prediction/background from oldEpochData to skip those requests.
                const [projResp, projNeighResp] = await Promise.all([
                    BackendAPI.fetchEpochProjection(contentPath, vis_method, currentVisID, targetEpoch, true),
                    BackendAPI.getProjectionNeighbors(contentPath, vis_method, currentVisID, targetEpoch, true),
                ]);

                const newEpochData = {
                    ...oldEpochData,
                    projection: projResp.projection || oldEpochData.projection,
                    projectionNeighbors: projNeighResp.neighbors || oldEpochData.projectionNeighbors,
                    indexList: projNeighResp.index_list || oldEpochData.indexList,
                };

                // Write updated epoch data to store so canvas re-renders
                const state = useGlobalStore.getState();
                state.setValue('allEpochData', { ...state.allEpochData, [targetEpoch]: newEpochData });

                const metrics = await evaluateProjectionQuality(epoch, selectedIndices, oldEpochData, newEpochData);
                if (metrics) {
                    useGlobalStore.getState().setValue('refineMetrics', {
                        focusDisplacement: metrics.avgFocusShift,
                        globalDrift: metrics.avgGlobalDrift,
                        neighborPreservation: metrics.avgNeighborConsistency,
                        trustworthiness: metrics.avgTrustworthiness,
                    });
                }
                calculateDisplacementStats(oldEpochData.projection, newEpochData.projection, selectedIndices, newEpochData.indexList || []);

                message.success({ content: `Epoch ${targetEpoch} refined! Plot updated!`, key: REFINE_MSG_KEY });
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
                {/* [逻辑更替]：不再监听模式改变自动触发，
                    而是将 handleUpdate 传给子组件，由子组件的 "Update" 按钮显式调用。
                */}
                <FunctionViewPanels onUpdateProjection={handleUpdate} />
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
export function FunctionViewPanels({ onUpdateProjection }: FunctionViewPanelsProps) {
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
                    <FunctionPanel onUpdateProjection={onUpdateProjection} />
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