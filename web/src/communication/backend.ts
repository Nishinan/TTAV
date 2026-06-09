import axios, { AxiosResponse } from 'axios';

// Backend server configuration
const DEFAULT_HOST = '';

interface NetworkOptions {
    host?: string;
}

export interface ViewportBBox {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
}

/**
 * Interfaces
 */
export interface BriefProjectionResult {
    proj: number[][];
    labels: number[];
    scale: number[];
}

/**
 * Basic HTTP request functions using axios
 */
function getFullUrl(path: string, options?: NetworkOptions): string {
    const host = options?.host || DEFAULT_HOST;
    return `${host}${path}`;
}

async function basicGetWithJsonResponse(path: string, options?: NetworkOptions): Promise<any> {
    try {
        const response: AxiosResponse = await axios.get(getFullUrl(path, options));
        return response.data;
    } catch (error) {
        throw new Error(`GET ${getFullUrl(path, options)} failed: ${error}`);
    }
}

async function basicPostWithJsonResponse(path: string, data: any, options?: NetworkOptions): Promise<any> {
    try {
        const response: AxiosResponse = await axios.post(getFullUrl(path, options), data, {
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        return response.data;
    } catch (error) {
        throw new Error(`POST ${getFullUrl(path, options)} failed: ${error}`);
    }
}

// code-philia/time-travelling-visualizer/time-travelling-visualizer-feat-embedding-atlas/web/src/communication/backend.ts

/**
 * Notify backend of current focus indices and chosen precision mode
 */
export function updateFocusContext(
    contentPath: string,
    selectedIndices: number[],
    focusMode: string,
    currentEpoch?: number,
    zoomBBox?: ViewportBBox | null,
    options?: NetworkOptions
) {
    const data: Record<string, any> = {
        "content_path": contentPath,
        "selected_indices": selectedIndices,
        "focus_mode": focusMode,
    };
    if (currentEpoch !== undefined) {
        data["current_epoch"] = currentEpoch;
    }
    if (zoomBBox) {
        data["zoom_bbox"] = {
            "x_min": zoomBBox.xMin,
            "x_max": zoomBBox.xMax,
            "y_min": zoomBBox.yMin,
            "y_max": zoomBBox.yMax,
        };
    }
    return basicPostWithJsonResponse('/updateFocusContext', data, options);
}

export function startRefineSession(
    contentPath: string,
    selectedIndices: number[],
    focusMode: string,
    currentEpoch?: number,
    zoomBBox?: ViewportBBox | null,
    options?: NetworkOptions
) {
    const data: Record<string, any> = {
        "content_path": contentPath,
        "selected_indices": selectedIndices,
        "focus_mode": focusMode,
    };
    if (currentEpoch !== undefined) {
        data["current_epoch"] = currentEpoch;
    }
    if (zoomBBox) {
        data["zoom_bbox"] = {
            "x_min": zoomBBox.xMin,
            "x_max": zoomBBox.xMax,
            "y_min": zoomBBox.yMin,
            "y_max": zoomBBox.yMax,
        };
    }
    return basicPostWithJsonResponse('/startRefineSession', data, options);
}

export function getRefineSessionProgress(
    sessionId: string,
    sinceVersion: number = -1,
    options?: NetworkOptions
) {
    return basicPostWithJsonResponse('/getRefineSessionProgress', {
        "session_id": sessionId,
        "since_version": sinceVersion,
    }, options);
}

export function stopRefineSession(
    sessionId: string,
    options?: NetworkOptions
) {
    return basicPostWithJsonResponse('/stopRefineSession', {
        "session_id": sessionId,
    }, options);
}

export async function syncSession(config: any): Promise<any> {
    // 这里的路由名需要和 Python server 中的 @app.route('/syncSession') 对应
    return basicPostWithJsonResponse('/syncSession', config);
}
/**
 * Backend API functions
 */
export function triggerStartVisualizing(
    contentPath: string, 
    visMethod: string, 
    visID: string, 
    dataType: string, 
    taskType: string,
    visConfig: any, 
    options?: NetworkOptions
) {
    const data = {
        "content_path": contentPath,
        "vis_method": visMethod,
        "vis_id": visID,
        "data_type": dataType,
        "task_type": taskType,
        "vis_config": visConfig
    };
    return basicPostWithJsonResponse('/startVisualizing', data, options);
}

export function fetchTrainingProcessInfo(contentPath: string, options?: NetworkOptions) {
    return basicGetWithJsonResponse(`/getTrainingProcessInfo?content_path=${encodeURIComponent(contentPath)}`, options);
}

export async function fetchEpochProjection(
    contentPath: string,
    vis_method: string,
    visID: string,
    epoch: number,
    refineFlag: boolean = false,
    options?: NetworkOptions
) {
    const data = {
        "content_path": contentPath,
        "vis_method": vis_method,
        "vis_id": visID,
        "epoch": `${epoch}`,
        "refine_flag": refineFlag,
    };
    return basicPostWithJsonResponse('/updateProjection', data, options);
}

export function getText(contentPath: string, options?: NetworkOptions) {
    const data = {
        "content_path": contentPath
    };
    return basicPostWithJsonResponse('/getAllText', data, options);
}

export function getAlignment(contentPath: string, options?: NetworkOptions) {
    const data = {
        "content_path": contentPath
    };
    return basicPostWithJsonResponse('/getAlignment', data, options);
}

export function getAttributeResource(
    contentPath: string, 
    epoch: number, 
    attributeName: string, 
    options?: NetworkOptions
) {
    const data = {
        "content_path": contentPath,
        "epoch": `${epoch}`,
        "attributes": [attributeName]
    };
    return basicPostWithJsonResponse('/getAttributes', data, options);
}

export function getOriginalNeighbors(contentPath: string, epoch: number, options?: NetworkOptions) {
    const data = {
        "content_path": contentPath,
        "epoch": epoch
    };
    return basicPostWithJsonResponse('/getOriginalNeighbors', data, options);
}

export function getProjectionNeighbors(
    contentPath: string,
    vis_method: string,
    vis_id: string,
    epoch: number,
    refineFlag: boolean = false,
    blendBBox?: ViewportBBox | null,
    blendFocusIndices?: number[] | null,
    blendDecayRatio?: number,
    options?: NetworkOptions
) {
    const data: Record<string, any> = {
        "content_path": contentPath,
        "vis_method": vis_method,
        "vis_id": vis_id,
        "epoch": epoch,
        "refine_flag": refineFlag
    };
    if (blendBBox) {
        data["blend_bbox"] = {
            "x_min": blendBBox.xMin,
            "x_max": blendBBox.xMax,
            "y_min": blendBBox.yMin,
            "y_max": blendBBox.yMax,
        };
    }
    if (blendFocusIndices && blendFocusIndices.length > 0) {
        data["blend_focus_indices"] = blendFocusIndices;
    }
    if (blendDecayRatio !== undefined) {
        data["blend_decay_ratio"] = blendDecayRatio;
    }
    return basicPostWithJsonResponse('/getProjectionNeighbors', data, options);
}

export function getBackground(
    contentPath: string, 
    vis_method: string,
    visID: string, 
    epoch: number | undefined, 
    options?: NetworkOptions
) {
    const data = {
        "content_path": contentPath,
        "vis_method": vis_method,
        "vis_id": visID,
        "epoch": `${epoch}`
    };
    return basicPostWithJsonResponse('/getBackground', data, options).then((response) => {
        const { background_image_base64 } = response as { background_image_base64: string };
        return `data:image/png;base64,${background_image_base64}`;
    });
}

export function getImageData(contentPath: string, index: number, options?: NetworkOptions) {
    const data = {
        "content_path": contentPath,
        "index": index
    };
    return basicPostWithJsonResponse('/getImageData', data, options).then((response) => {
        const { image_base64 } = response as { image_base64: string };
        return `data:image/png;base64,${image_base64}`;
    });
}

export function getTextData(contentPath: string, index: number, options?: NetworkOptions) {
    const data = {
        "content_path": contentPath,
        "index": index
    };
    return basicPostWithJsonResponse('/getTextData', data, options).then((response) => {
        const { text } = response as { text: string };
        return text;
    });
}

export function getVisualizeMetrics(
    contentPath: string, 
    vis_method: string,
    visID: string, 
    epoch: number, 
    options?: NetworkOptions
) {
    const data = {
        "content_path": contentPath,
        "vis_method": vis_method,
        "vis_id": visID,
        "epoch": `${epoch}`
    };
    return basicPostWithJsonResponse('/getVisualizeMetrics', data, options);
}

export function getRefineMetrics(
    contentPath: string,
    vis_method: string,
    vis_id: string,
    epoch: number,
    focusIndices: number[],
    refineFlag: boolean = false,
    blendBBox?: ViewportBBox | null,
    blendFocusIndices?: number[] | null,
    blendDecayRatio?: number,
    options?: NetworkOptions
) {
    const data: Record<string, any> = {
        "content_path": contentPath,
        "vis_method": vis_method,
        "vis_id": vis_id,
        "epoch": epoch,
        "focus_indices": focusIndices,
        "refine_flag": refineFlag,
    };
    if (blendBBox) {
        data["blend_bbox"] = {
            "x_min": blendBBox.xMin,
            "x_max": blendBBox.xMax,
            "y_min": blendBBox.yMin,
            "y_max": blendBBox.yMax,
        };
    }
    if (blendFocusIndices && blendFocusIndices.length > 0) {
        data["blend_focus_indices"] = blendFocusIndices;
    }
    if (blendDecayRatio !== undefined) {
        data["blend_decay_ratio"] = blendDecayRatio;
    }
    return basicPostWithJsonResponse('/getRefineMetrics', data, options);
}

export function getInfluenceSamples(
    contentPath: string,  
    epoch: number, 
    trainingEvent: any, 
    options?: NetworkOptions
) {
    const data = {
        "content_path": contentPath,
        "epoch": `${epoch}`,
        "training_event": trainingEvent,
        "num_samples": 10 // Default number of samples to fetch
    };
    return basicPostWithJsonResponse('/getInfluenceSamples', data, options);
}

export function calculateTrainingEvents(
    contentPath: string, 
    epoch: number, 
    eventTypes: string[], 
    options?: NetworkOptions
) {
    const data = {
        "content_path": contentPath,
        "epoch": `${epoch}`,
        "event_types": eventTypes
    };
    return basicPostWithJsonResponse('/calculateTrainingEvents', data, options);
}

// Test connection function
export function testConnection(message: string, options?: NetworkOptions) {
    const data = { message };
    return basicPostWithJsonResponse('/testConnection', data, options);
}
