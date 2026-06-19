import { useEffect, useMemo, useRef, useState } from 'react'
import { useDefaultStore } from '../state/state.unified';
import ChartComponent from './chart';
import { notifyEpochSwitch } from '../communication/extension';
import { Tooltip } from 'antd';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';

function useStableEpochs(epochs: number[]) {
    const ref = useRef<number[]>([]);
    if (
        epochs.length !== ref.current.length ||
        epochs.some((v, i) => v !== ref.current[i])
    ) {
        ref.current = epochs;
    }
    return ref.current;
}

interface TimelineProps {
    epoch: number;
    epochs: number[];
    progress: number;
    onSwitchEpoch: (epoch: number) => void;
}

function Timeline({ epoch, epochs, progress, onSwitchEpoch }: TimelineProps) {
    const stableEpochs = useStableEpochs(epochs);
    const [isPlaying, setIsPlaying] = useState(false);
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const currentEpochIndexRef = useRef<number>(stableEpochs.indexOf(epoch));

    const nodeOffset = 40;
    const NODE_LINE_HEIGHT = 60;
    const NODE_CENTER_Y = NODE_LINE_HEIGHT / 2;

    useEffect(() => {
        if (stableEpochs.length > 0) {
            onSwitchEpoch(stableEpochs[0]);
            currentEpochIndexRef.current = 0;
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [stableEpochs]);

    useEffect(() => {
        currentEpochIndexRef.current = stableEpochs.indexOf(epoch);
    }, [stableEpochs, epoch]);

    // Keyboard ← → navigation
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            const currentIndex = stableEpochs.indexOf(epoch);
            if (event.key === 'ArrowRight' && currentIndex < stableEpochs.length - 1) {
                onSwitchEpoch(stableEpochs[currentIndex + 1]);
            } else if (event.key === 'ArrowLeft' && currentIndex > 0) {
                onSwitchEpoch(stableEpochs[currentIndex - 1]);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [stableEpochs, epoch, onSwitchEpoch]);

    // Autoplay at fixed 1s interval
    const togglePlayPause = () => {
        if (isPlaying) {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        } else {
            intervalRef.current = setInterval(() => {
                const nextIndex = (currentEpochIndexRef.current + 1) % stableEpochs.length;
                if (nextIndex === 0) {
                    clearInterval(intervalRef.current!);
                    intervalRef.current = null;
                    setIsPlaying(false);
                } else {
                    onSwitchEpoch(stableEpochs[nextIndex]);
                    currentEpochIndexRef.current = nextIndex;
                }
            }, 1000);
        }
        setIsPlaying(!isPlaying);
    };

    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, []);

    const currentIndex = stableEpochs.indexOf(epoch);

    const nodes = useMemo(() => {
        return stableEpochs.map((e, index) => ({
            value: e,
            x: index * 40 + nodeOffset,
            y: NODE_CENTER_Y,
        }));
    }, [stableEpochs]);

    const svgWidth = useMemo(() => {
        if (nodes.length === 0) return 0;
        const maxX = Math.max(...nodes.map(n => n.x));
        return maxX + nodeOffset + 20;
    }, [nodes]);

    return (
        <div style={{ display: 'flex', alignItems: 'center', height: '100%', width: '100%', gap: 6, padding: '0 8px', boxSizing: 'border-box' }}>

            {/* Step back */}
            <Tooltip title="Previous epoch (←)">
                <button
                    onClick={() => { if (currentIndex > 0) onSwitchEpoch(stableEpochs[currentIndex - 1]); }}
                    disabled={currentIndex <= 0}
                    style={ctrlBtnStyle(currentIndex <= 0)}
                    aria-label="Previous epoch"
                >
                    <SkipBack size={12} />
                </button>
            </Tooltip>

            {/* Play / Pause */}
            <Tooltip title={isPlaying ? 'Pause' : 'Play'}>
                <button
                    onClick={togglePlayPause}
                    style={ctrlBtnStyle(false, true)}
                    aria-label={isPlaying ? 'Pause' : 'Play'}
                >
                    {isPlaying ? <Pause size={12} /> : <Play size={12} />}
                </button>
            </Tooltip>

            {/* Step forward */}
            <Tooltip title="Next epoch (→)">
                <button
                    onClick={() => { if (currentIndex < stableEpochs.length - 1) onSwitchEpoch(stableEpochs[currentIndex + 1]); }}
                    disabled={currentIndex >= stableEpochs.length - 1}
                    style={ctrlBtnStyle(currentIndex >= stableEpochs.length - 1)}
                    aria-label="Next epoch"
                >
                    <SkipForward size={12} />
                </button>
            </Tooltip>

            {/* SVG dot timeline (original style) */}
            <div style={{ flex: 1, overflowX: 'auto', overflowY: 'hidden', minWidth: 0 }}>
                <svg width={svgWidth} height={NODE_LINE_HEIGHT} className="timeline-svg">
                    <g transform="translate(20, 0)">
                        {/* Connecting lines */}
                        {nodes.map((node, index) => {
                            if (index >= nodes.length - 1) return null;
                            const next = nodes[index + 1];
                            const loadedCount = (progress / 100) * nodes.length;
                            const isLoaded = loadedCount >= (index + 2);
                            return (
                                <line
                                    key={`link-${index}`}
                                    x1={node.x} y1={node.y}
                                    x2={next.x}  y2={next.y}
                                    stroke={isLoaded ? 'var(--accent-blue-light, #72A8F0)' : 'var(--layout-border-color, #e0e0e0)'}
                                    strokeWidth="1"
                                    style={{ transition: 'stroke 0.5s ease-in-out', strokeLinecap: 'round' }}
                                />
                            );
                        })}

                        {/* Epoch nodes */}
                        {nodes.map((node, index) => {
                            const loadedCount = (progress / 100) * nodes.length;
                            const isLoaded = loadedCount >= (index + 1);
                            const isActive = node.value === epoch;
                            const fill = isLoaded
                                ? (isActive ? 'var(--accent-blue, #3278F0)' : 'var(--accent-blue-light, #72A8F0)')
                                : 'var(--layout-border-color, #e0e0e0)';
                            return (
                                <g key={index} transform={`translate(${node.x}, ${node.y})`}>
                                    <circle
                                        r="8"
                                        fill={fill}
                                        stroke={fill}
                                        className="timeline-node"
                                        style={{ transition: 'all 0.5s ease-in-out', cursor: 'pointer' }}
                                        onClick={() => onSwitchEpoch(node.value)}
                                    />
                                    <text
                                        x="0" y="-14"
                                        style={{
                                            fill,
                                            transition: 'fill 0.5s ease-in-out',
                                            fontSize: '12px',
                                            userSelect: 'none',
                                        }}
                                        textAnchor="middle"
                                    >
                                        {node.value}
                                    </text>
                                </g>
                            );
                        })}
                    </g>
                </svg>
            </div>
        </div>
    );
}

function ctrlBtnStyle(disabled: boolean, primary = false): React.CSSProperties {
    return {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 24,
        height: 24,
        borderRadius: 4,
        border: `1px solid ${primary ? 'var(--accent-blue, #3278F0)' : 'var(--layout-border-color, #ccc)'}`,
        background: primary ? 'var(--accent-blue, #3278F0)' : 'var(--surface-color, #fff)',
        color: primary ? '#fff' : disabled ? 'var(--text-muted, #ccc)' : 'var(--text-primary, #333)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.4 : 1,
        flexShrink: 0,
        transition: 'background 0.15s, border-color 0.15s',
        padding: 0,
    };
}

export function MainBlock() {
    const { epoch, setEpoch } = useDefaultStore(['epoch', 'setEpoch']);
    const { availableEpochs } = useDefaultStore(['availableEpochs']);
    const { progress } = useDefaultStore(['progress']);

    return (
        <div className="canvas-column">
            <ChartComponent />
            <div id="footer">
                <Timeline
                    epoch={epoch}
                    epochs={availableEpochs}
                    progress={progress}
                    onSwitchEpoch={(e) => {
                        setEpoch(e);
                        notifyEpochSwitch(e);
                    }}
                />
            </div>
        </div>
    );
}
