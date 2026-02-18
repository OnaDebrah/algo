import React, { useState, useRef, useCallback, useMemo } from 'react';
import {
    Plus,
    Trash2,
    Settings2,
    Play,
    Save,
    Zap,
    Target,
    BrainCircuit,
    Filter,
    ArrowRight,
    ChevronRight,
    Search,
    AlertCircle,
    Link2,
    X,
    Eye
} from 'lucide-react';
import { VisualBlock, BlockType, MLModel } from '@/types/all_types';

/* ─── Types ───────────────────────────────────────────────────────── */

interface Connection {
    from: string;   // source block id (output port)
    to: string;     // target block id (input port)
}

interface ValidationError {
    blockId: string;
    message: string;
}

interface StrategyBuilderProps {
    models: MLModel[];
    isLoading?: boolean;
    onSave?: (config: { blocks: VisualBlock[]; connections: Connection[]; root_block_id: string }) => void;
}

/* ─── Block port geometry helpers ─────────────────────────────────── */

function getOutputPortPos(block: VisualBlock): { x: number; y: number } {
    const x = (block.position?.x ?? 0) + 180;  // right edge
    const y = (block.position?.y ?? 0) + 40;    // vertical center
    return { x, y };
}

function getInputPortPos(block: VisualBlock): { x: number; y: number } {
    const x = (block.position?.x ?? 0);         // left edge
    const y = (block.position?.y ?? 0) + 40;    // vertical center
    return { x, y };
}

/* ─── Component ───────────────────────────────────────────────────── */

const StrategyBuilder: React.FC<StrategyBuilderProps> = ({ models, isLoading = false, onSave }) => {
    /* State */
    const [blocks, setBlocks] = useState<VisualBlock[]>([
        { id: 'root', type: 'output', label: 'Final Signal', params: { label: 'Final Signal' }, position: { x: 700, y: 220 } }
    ]);
    const [connections, setConnections] = useState<Connection[]>([]);
    const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState('');

    // Drag state (refs to avoid re-renders during drag)
    const dragRef = useRef<{ blockId: string; offsetX: number; offsetY: number } | null>(null);
    const canvasRef = useRef<HTMLDivElement>(null);

    // Connection drawing state
    const [connectingFrom, setConnectingFrom] = useState<string | null>(null);
    const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

    // Preview state
    const [showPreview, setShowPreview] = useState(false);

    // Validation errors
    const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);

    /* Block library */
    const blockLibrary = [
        { type: 'indicator' as BlockType, category: 'Signals', label: 'RSI', icon: <Filter size={16} />, description: 'Relative Strength Index', defaultParams: { type: 'rsi', period: 14, op: '>', val: 70 } },
        { type: 'indicator' as BlockType, category: 'Signals', label: 'SMA', icon: <ChevronRight size={16} />, description: 'Simple Moving Average', defaultParams: { type: 'sma', period: 20, op: 'cross_above', val: 0 } },
        { type: 'indicator' as BlockType, category: 'Signals', label: 'MACD', icon: <ChevronRight size={16} />, description: 'Moving Average Conv/Div', defaultParams: { type: 'macd', period: 12, op: 'cross_above', val: 0 } },
        { type: 'ml_model' as BlockType, category: 'AI Models', label: 'Trained Model', icon: <BrainCircuit size={16} />, description: 'Neural Network / RF', defaultParams: {} },
        { type: 'logic' as BlockType, category: 'Logic', label: 'AND Gate', icon: <ArrowRight size={16} />, description: 'All signals must match', defaultParams: { op: 'AND', inputs: [] } },
        { type: 'logic' as BlockType, category: 'Logic', label: 'OR Gate', icon: <Plus size={16} />, description: 'Any signal can match', defaultParams: { op: 'OR', inputs: [] } },
        { type: 'risk' as BlockType, category: 'Management', label: 'Take Profit', icon: <Target size={16} />, description: 'Exit at target', defaultParams: { risk_type: 'take_profit', threshold: 5 } },
        { type: 'risk' as BlockType, category: 'Management', label: 'Stop Loss', icon: <Target size={16} />, description: 'Exit at loss', defaultParams: { risk_type: 'stop_loss', threshold: 3 } },
    ];

    /* ── Helpers ───────────────────────────────────────────────────── */

    const blocksById = useMemo(() => {
        const map: Record<string, VisualBlock> = {};
        blocks.forEach(b => { map[b.id] = b; });
        return map;
    }, [blocks]);

    const getBlockError = (blockId: string) => validationErrors.find(e => e.blockId === blockId);

    const getConnectionsTo = (blockId: string) => connections.filter(c => c.to === blockId);
    const getConnectionsFrom = (blockId: string) => connections.filter(c => c.from === blockId);

    /* ── Add block ────────────────────────────────────────────────── */

    const addBlock = (item: typeof blockLibrary[0]) => {
        const newBlock: VisualBlock = {
            id: `b-${Date.now()}`,
            type: item.type,
            label: item.label,
            params: { ...item.defaultParams },
            position: { x: 80 + Math.random() * 200, y: 80 + Math.random() * 300 }
        };
        setBlocks(prev => [...prev, newBlock]);
        setSelectedBlockId(newBlock.id);
        setValidationErrors([]);
    };

    /* ── Delete block ─────────────────────────────────────────────── */

    const deleteBlock = (id: string) => {
        if (id === 'root') return;
        // Remove block and all connections involving it
        setBlocks(prev => prev.filter(b => b.id !== id));
        setConnections(prev => {
            const removed = prev.filter(c => c.from === id || c.to === id);
            const remaining = prev.filter(c => c.from !== id && c.to !== id);
            // Clean up params.inputs on logic/output/risk blocks that lost a connection
            removed.forEach(conn => {
                setBlocks(current => current.map(b => {
                    if (b.id === conn.to) {
                        if (b.type === 'logic') {
                            return { ...b, params: { ...b.params, inputs: (b.params.inputs || []).filter((i: string) => i !== conn.from) } };
                        }
                        if (b.type === 'output' || b.type === 'risk') {
                            if (b.params.input === conn.from) {
                                return { ...b, params: { ...b.params, input: undefined } };
                            }
                        }
                    }
                    return b;
                }));
            });
            return remaining;
        });
        setSelectedBlockId(null);
        setValidationErrors([]);
    };

    /* ── Drag handling ────────────────────────────────────────────── */

    const handleBlockMouseDown = (e: React.MouseEvent, blockId: string) => {
        // Don't start drag if clicking on a port
        if ((e.target as HTMLElement).closest('[data-port]')) return;
        e.stopPropagation();
        e.preventDefault();
        const block = blocksById[blockId];
        if (!block) return;

        const canvasRect = canvasRef.current?.getBoundingClientRect();
        if (!canvasRect) return;

        dragRef.current = {
            blockId,
            offsetX: e.clientX - canvasRect.left - (block.position?.x ?? 0),
            offsetY: e.clientY - canvasRect.top - (block.position?.y ?? 0),
        };
        setSelectedBlockId(blockId);
    };

    const handleCanvasMouseMove = useCallback((e: React.MouseEvent) => {
        const canvasRect = canvasRef.current?.getBoundingClientRect();
        if (!canvasRect) return;

        const mx = e.clientX - canvasRect.left;
        const my = e.clientY - canvasRect.top;

        // Track mouse for connection drawing
        if (connectingFrom) {
            setMousePos({ x: mx, y: my });
        }

        // Block dragging
        if (dragRef.current) {
            const { blockId, offsetX, offsetY } = dragRef.current;
            const newX = Math.max(0, mx - offsetX);
            const newY = Math.max(0, my - offsetY);
            setBlocks(prev => prev.map(b =>
                b.id === blockId ? { ...b, position: { x: newX, y: newY } } : b
            ));
        }
    }, [connectingFrom]);

    const handleCanvasMouseUp = useCallback(() => {
        dragRef.current = null;
        if (connectingFrom) {
            // Dropped in empty space — cancel connection
            setConnectingFrom(null);
        }
    }, [connectingFrom]);

    /* ── Connection handling ──────────────────────────────────────── */

    const handleOutputPortClick = (e: React.MouseEvent, blockId: string) => {
        e.stopPropagation();
        e.preventDefault();
        setConnectingFrom(blockId);
    };

    const handleInputPortClick = (e: React.MouseEvent, targetBlockId: string) => {
        e.stopPropagation();
        e.preventDefault();

        if (!connectingFrom || connectingFrom === targetBlockId) {
            setConnectingFrom(null);
            return;
        }

        // Prevent duplicate connections
        const exists = connections.some(c => c.from === connectingFrom && c.to === targetBlockId);
        if (exists) {
            setConnectingFrom(null);
            return;
        }

        const targetBlock = blocksById[targetBlockId];
        if (!targetBlock) {
            setConnectingFrom(null);
            return;
        }

        // For output/risk blocks, only allow one input — replace existing
        if (targetBlock.type === 'output' || targetBlock.type === 'risk') {
            setConnections(prev => [...prev.filter(c => c.to !== targetBlockId), { from: connectingFrom, to: targetBlockId }]);
            setBlocks(prev => prev.map(b =>
                b.id === targetBlockId ? { ...b, params: { ...b.params, input: connectingFrom } } : b
            ));
        }
        // For logic blocks, push to inputs array
        else if (targetBlock.type === 'logic') {
            const currentInputs: string[] = targetBlock.params.inputs || [];
            if (!currentInputs.includes(connectingFrom)) {
                setConnections(prev => [...prev, { from: connectingFrom, to: targetBlockId }]);
                setBlocks(prev => prev.map(b =>
                    b.id === targetBlockId ? { ...b, params: { ...b.params, inputs: [...currentInputs, connectingFrom] } } : b
                ));
            }
        }

        setConnectingFrom(null);
        setValidationErrors([]);
    };

    const removeConnection = (from: string, to: string) => {
        setConnections(prev => prev.filter(c => !(c.from === from && c.to === to)));
        // Clean up target block's params
        setBlocks(prev => prev.map(b => {
            if (b.id === to) {
                if (b.type === 'logic') {
                    return { ...b, params: { ...b.params, inputs: (b.params.inputs || []).filter((i: string) => i !== from) } };
                }
                if ((b.type === 'output' || b.type === 'risk') && b.params.input === from) {
                    return { ...b, params: { ...b.params, input: undefined } };
                }
            }
            return b;
        }));
    };

    /* ── Validation ───────────────────────────────────────────────── */

    const validate = (): boolean => {
        const errors: ValidationError[] = [];

        const rootBlock = blocksById['root'];
        if (rootBlock && !rootBlock.params.input) {
            errors.push({ blockId: 'root', message: 'Connect a block to the output' });
        }

        blocks.forEach(b => {
            if (b.type === 'logic') {
                const inputs = b.params.inputs || [];
                if (inputs.length === 0) {
                    errors.push({ blockId: b.id, message: 'Connect at least 1 input' });
                }
            }
            if (b.type === 'ml_model' && !b.params.model_id) {
                errors.push({ blockId: b.id, message: 'Select a trained model' });
            }
        });

        setValidationErrors(errors);
        return errors.length === 0;
    };

    /* ── Preview ──────────────────────────────────────────────────── */

    const buildPreviewText = (): string => {
        const visited = new Set<string>();
        const describe = (id: string): string => {
            if (visited.has(id)) return '...';
            visited.add(id);
            const b = blocksById[id];
            if (!b) return '?';

            if (b.type === 'indicator') {
                const t = (b.params.type || 'rsi').toUpperCase();
                const p = b.params.period || 14;
                const op = b.params.op || '>';
                const v = b.params.val || 50;
                if (t === 'MACD') return `MACD(${op.replace('_', ' ')})`;
                if (t === 'SMA') return `Close ${op.replace('_', ' ')} SMA(${p})`;
                return `${t}(${p}) ${op} ${v}`;
            }
            if (b.type === 'ml_model') {
                const mid = b.params.model_id;
                return mid ? `ML(${mid.slice(0, 12)})` : 'ML(?)';
            }
            if (b.type === 'logic') {
                const op = b.params.op || 'AND';
                const inputs = (b.params.inputs || []).map((i: string) => describe(i));
                return `(${inputs.join(` ${op} `)})`;
            }
            if (b.type === 'risk') {
                const rt = b.params.risk_type || 'stop_loss';
                const th = b.params.threshold || 0;
                const child = b.params.input ? describe(b.params.input) : '?';
                return `${child} → ${rt === 'take_profit' ? 'TP' : 'SL'}(${th}%)`;
            }
            if (b.type === 'output') {
                const child = b.params.input ? describe(b.params.input) : '(empty)';
                return `${child} → Signal`;
            }
            return b.label || b.type;
        };
        return describe('root');
    };

    /* ── Save / Connect Engine ────────────────────────────────────── */

    const handleSave = () => {
        if (!validate()) return;

        // Enrich blocks: make sure params.inputs / params.input are set from connections
        const enrichedBlocks = blocks.map(b => {
            const incomingConns = connections.filter(c => c.to === b.id);
            if (b.type === 'logic') {
                return { ...b, params: { ...b.params, inputs: incomingConns.map(c => c.from) } };
            }
            if (b.type === 'output' || b.type === 'risk') {
                return { ...b, params: { ...b.params, input: incomingConns[0]?.from || undefined } };
            }
            return b;
        });

        onSave?.({ blocks: enrichedBlocks, connections, root_block_id: 'root' });
    };

    /* ── Rendering helpers ────────────────────────────────────────── */

    const selectedBlock = blocks.find(b => b.id === selectedBlockId);

    const getBlockColor = (type: BlockType) => {
        switch (type) {
            case 'ml_model': return { bg: 'bg-fuchsia-500/20', text: 'text-fuchsia-400', border: 'border-fuchsia-500' };
            case 'indicator': return { bg: 'bg-indigo-500/20', text: 'text-indigo-400', border: 'border-indigo-500' };
            case 'logic': return { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500' };
            case 'risk': return { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500' };
            case 'output': return { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500' };
            default: return { bg: 'bg-slate-500/20', text: 'text-slate-400', border: 'border-slate-500' };
        }
    };

    const getBlockIcon = (type: BlockType) => {
        switch (type) {
            case 'ml_model': return <BrainCircuit size={14} />;
            case 'indicator': return <Zap size={14} />;
            case 'logic': return <ArrowRight size={14} />;
            case 'risk': return <Target size={14} />;
            case 'output': return <Play size={14} />;
            default: return <Zap size={14} />;
        }
    };

    const getBlockDescription = (block: VisualBlock) => {
        if (block.params.model_id) return `model: ${block.params.model_id.slice(0, 10)}…`;
        if (block.type === 'indicator') {
            const t = (block.params.type || 'rsi').toUpperCase();
            if (t === 'MACD') return `MACD signal`;
            return `${t}(${block.params.period || 14}) ${block.params.op || '>'} ${block.params.val || 50}`;
        }
        if (block.type === 'logic') return `${block.params.op || 'AND'} · ${(block.params.inputs || []).length} inputs`;
        if (block.type === 'risk') return `${block.params.risk_type === 'take_profit' ? 'TP' : 'SL'} ${block.params.threshold || 0}%`;
        if (block.type === 'output') return block.params.input ? 'Connected' : 'Not connected';
        return 'Unconfigured';
    };

    /* ── SVG connection curves ────────────────────────────────────── */

    const renderConnections = () => {
        const lines: React.ReactNode[] = [];

        connections.forEach((conn, idx) => {
            const fromBlock = blocksById[conn.from];
            const toBlock = blocksById[conn.to];
            if (!fromBlock || !toBlock) return;

            const start = getOutputPortPos(fromBlock);
            const end = getInputPortPos(toBlock);
            const dx = Math.abs(end.x - start.x) * 0.5;

            lines.push(
                <path
                    key={`conn-${idx}`}
                    d={`M ${start.x} ${start.y} C ${start.x + dx} ${start.y}, ${end.x - dx} ${end.y}, ${end.x} ${end.y}`}
                    stroke="rgba(217, 70, 239, 0.5)"
                    strokeWidth={2}
                    fill="none"
                    className="transition-all"
                />
            );
        });

        // In-progress connection line
        if (connectingFrom) {
            const fromBlock = blocksById[connectingFrom];
            if (fromBlock) {
                const start = getOutputPortPos(fromBlock);
                const dx = Math.abs(mousePos.x - start.x) * 0.5;
                lines.push(
                    <path
                        key="connecting"
                        d={`M ${start.x} ${start.y} C ${start.x + dx} ${start.y}, ${mousePos.x - dx} ${mousePos.y}, ${mousePos.x} ${mousePos.y}`}
                        stroke="rgba(217, 70, 239, 0.3)"
                        strokeWidth={2}
                        strokeDasharray="6 4"
                        fill="none"
                    />
                );
            }
        }

        return (
            <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 1 }}>
                {lines}
            </svg>
        );
    };

    /* ═══════════════════════════════════════════════════════════════ */
    /* ── RENDER ────────────────────────────────────────────────────── */
    /* ═══════════════════════════════════════════════════════════════ */

    return (
        <div className="flex h-[700px] bg-slate-950/20 border border-slate-800 rounded-3xl overflow-hidden animate-in fade-in zoom-in-95 duration-500">
            {/* ── Library Sidebar ──────────────────────────────────── */}
            <div className="w-72 border-r border-slate-800 p-5 bg-slate-900/40 flex flex-col">
                <div className="mb-5">
                    <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                        <Plus className="text-fuchsia-400" size={18} />
                        Components
                    </h3>
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={13} />
                        <input
                            type="text"
                            placeholder="Search blocks..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full bg-slate-950/50 border border-slate-700/50 rounded-xl py-2 pl-9 pr-4 text-xs text-slate-300 focus:border-fuchsia-500 outline-none transition-all"
                        />
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto space-y-5">
                    {['Signals', 'AI Models', 'Logic', 'Management'].map(cat => {
                        const items = blockLibrary.filter(b => b.category === cat && b.label.toLowerCase().includes(searchTerm.toLowerCase()));
                        if (items.length === 0) return null;
                        return (
                            <div key={cat}>
                                <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">{cat}</h4>
                                <div className="space-y-1.5">
                                    {items.map((item, idx) => (
                                        <button
                                            key={idx}
                                            onClick={() => addBlock(item)}
                                            className="w-full p-2.5 bg-slate-950/30 border border-slate-800 hover:border-slate-600 rounded-xl flex items-center gap-3 group transition-all"
                                        >
                                            <div className="p-1.5 bg-slate-900 rounded-lg group-hover:bg-slate-800 text-slate-400 group-hover:text-fuchsia-400 transition-all">
                                                {item.icon}
                                            </div>
                                            <div className="text-left">
                                                <div className="text-xs font-bold text-slate-200">{item.label}</div>
                                                <div className="text-[10px] text-slate-500 leading-tight">{item.description}</div>
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* Connection mode hint */}
                {connectingFrom && (
                    <div className="mt-3 p-3 bg-fuchsia-500/10 border border-fuchsia-500/30 rounded-xl animate-in fade-in">
                        <div className="flex items-center gap-2 text-xs text-fuchsia-400 font-medium">
                            <Link2 size={14} className="animate-pulse" />
                            Click an input port to connect
                        </div>
                        <button
                            onClick={() => setConnectingFrom(null)}
                            className="mt-2 text-[10px] text-slate-500 hover:text-slate-300 flex items-center gap-1"
                        >
                            <X size={10} /> Cancel
                        </button>
                    </div>
                )}
            </div>

            {/* ── Canvas Workspace ─────────────────────────────────── */}
            <div
                ref={canvasRef}
                className={`flex-1 relative bg-slate-950/50 overflow-hidden ${connectingFrom ? 'cursor-crosshair' : 'cursor-default'}`}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                onClick={() => { setSelectedBlockId(null); if (connectingFrom) setConnectingFrom(null); }}
            >
                {/* Grid bg */}
                <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(#6366f1 1px, transparent 0)', backgroundSize: '24px 24px' }} />

                {/* SVG connections layer */}
                {renderConnections()}

                {/* Blocks on canvas */}
                {blocks.map(block => {
                    const colors = getBlockColor(block.type);
                    const error = getBlockError(block.id);
                    const isSelected = selectedBlockId === block.id;

                    return (
                        <div
                            key={block.id}
                            onMouseDown={(e) => handleBlockMouseDown(e, block.id)}
                            onClick={(e) => { e.stopPropagation(); setSelectedBlockId(block.id); }}
                            className={`absolute p-3.5 min-w-[180px] rounded-2xl border-2 transition-colors shadow-2xl select-none ${
                                error
                                    ? 'bg-red-950/60 border-red-500 ring-2 ring-red-500/20'
                                    : isSelected
                                        ? `bg-slate-900 ${colors.border} ring-4 ring-fuchsia-500/10`
                                        : 'bg-slate-900/90 border-slate-800 hover:border-slate-700'
                            }`}
                            style={{ left: block.position?.x, top: block.position?.y, zIndex: isSelected ? 20 : 10, cursor: dragRef.current?.blockId === block.id ? 'grabbing' : 'grab' }}
                        >
                            {/* Header */}
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    <div className={`p-1.5 rounded-lg ${colors.bg} ${colors.text}`}>
                                        {getBlockIcon(block.type)}
                                    </div>
                                    <span className="text-xs font-bold text-white">{block.label || block.type}</span>
                                </div>
                            </div>

                            {/* Description */}
                            <div className="text-[10px] text-slate-400 font-mono overflow-hidden truncate italic opacity-60">
                                {getBlockDescription(block)}
                            </div>

                            {/* Error badge */}
                            {error && (
                                <div className="mt-2 text-[10px] text-red-400 flex items-center gap-1">
                                    <AlertCircle size={10} />
                                    {error.message}
                                </div>
                            )}

                            {/* Output port (right) — click to start connection */}
                            {block.type !== 'output' && (
                                <div
                                    data-port="output"
                                    onClick={(e) => handleOutputPortClick(e, block.id)}
                                    className={`absolute top-1/2 -right-2 h-4 w-4 rounded-full border-2 border-slate-950 cursor-pointer transition-all hover:scale-125 ${
                                        connectingFrom === block.id ? 'bg-fuchsia-400 ring-2 ring-fuchsia-400/40' : 'bg-fuchsia-500'
                                    }`}
                                    style={{ transform: 'translateY(-50%)' }}
                                />
                            )}

                            {/* Input port (left) — click to complete connection */}
                            {block.type !== 'output' ? (
                                <div
                                    data-port="input"
                                    onClick={(e) => handleInputPortClick(e, block.id)}
                                    className={`absolute top-1/2 -left-2 h-4 w-4 rounded-full border-2 border-slate-950 cursor-pointer transition-all hover:scale-125 ${
                                        connectingFrom ? 'bg-fuchsia-400 ring-2 ring-fuchsia-400/40 animate-pulse' : 'bg-fuchsia-500/40'
                                    }`}
                                    style={{ transform: 'translateY(-50%)' }}
                                />
                            ) : (
                                /* Output block only has an input port */
                                <div
                                    data-port="input"
                                    onClick={(e) => handleInputPortClick(e, block.id)}
                                    className={`absolute top-1/2 -left-2 h-4 w-4 rounded-full border-2 border-slate-950 cursor-pointer transition-all hover:scale-125 ${
                                        connectingFrom ? 'bg-emerald-400 ring-2 ring-emerald-400/40 animate-pulse' : 'bg-emerald-500'
                                    }`}
                                    style={{ transform: 'translateY(-50%)' }}
                                />
                            )}
                        </div>
                    );
                })}

                {/* Canvas overlay controls */}
                <div className="absolute top-5 right-5 flex gap-2" style={{ zIndex: 30 }}>
                    <button
                        onClick={(e) => { e.stopPropagation(); setShowPreview(!showPreview); }}
                        className="px-4 py-2 bg-slate-800/80 backdrop-blur hover:bg-slate-700 rounded-xl text-slate-300 text-sm font-medium flex items-center gap-2 border border-slate-700 transition-all shadow-lg"
                    >
                        <Eye size={16} />
                        Preview
                    </button>
                    <button
                        onClick={(e) => { e.stopPropagation(); handleSave(); }}
                        className="px-4 py-2 bg-fuchsia-600 hover:bg-fuchsia-500 rounded-xl text-white text-sm font-bold flex items-center gap-2 transition-all shadow-lg shadow-fuchsia-600/20"
                    >
                        <Save size={16} />
                        Connect Engine
                    </button>
                </div>

                {/* Preview toast */}
                {showPreview && (
                    <div className="absolute bottom-5 left-5 right-5 bg-slate-900/95 backdrop-blur border border-slate-700 rounded-2xl p-4 shadow-2xl animate-in slide-in-from-bottom-4" style={{ zIndex: 30 }}>
                        <div className="flex items-center justify-between mb-2">
                            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                                <Eye size={14} className="text-fuchsia-400" />
                                Strategy Preview
                            </h4>
                            <button onClick={() => setShowPreview(false)} className="text-slate-500 hover:text-slate-300">
                                <X size={14} />
                            </button>
                        </div>
                        <p className="text-sm text-fuchsia-300 font-mono">{buildPreviewText()}</p>
                        <p className="text-[10px] text-slate-500 mt-2">{blocks.length - 1} blocks · {connections.length} connections</p>
                    </div>
                )}

                {/* Validation errors summary */}
                {validationErrors.length > 0 && (
                    <div className="absolute bottom-5 left-5 bg-red-950/90 backdrop-blur border border-red-500/30 rounded-xl p-3 shadow-xl animate-in fade-in" style={{ zIndex: 30 }}>
                        <div className="flex items-center gap-2 mb-1">
                            <AlertCircle size={14} className="text-red-400" />
                            <span className="text-xs font-bold text-red-400">{validationErrors.length} issue{validationErrors.length > 1 ? 's' : ''} found</span>
                        </div>
                        {validationErrors.map((err, i) => (
                            <p key={i} className="text-[10px] text-red-400/70 ml-5">• {blocksById[err.blockId]?.label || err.blockId}: {err.message}</p>
                        ))}
                    </div>
                )}
            </div>

            {/* ── Properties Panel ─────────────────────────────────── */}
            <div className="w-72 border-l border-slate-800 p-5 bg-slate-900/40 flex flex-col overflow-y-auto">
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-5 flex items-center gap-2">
                    <Settings2 size={14} />
                    Properties
                </h3>

                {selectedBlock ? (
                    <div className="space-y-5 animate-in slide-in-from-right-4 duration-300">
                        {/* Label */}
                        <div className="space-y-1.5">
                            <label className="text-[10px] font-medium text-slate-500 uppercase">Label</label>
                            <input
                                type="text"
                                value={selectedBlock.label || ''}
                                onChange={(e) => setBlocks(prev => prev.map(b => b.id === selectedBlock.id ? { ...b, label: e.target.value } : b))}
                                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-sm text-white outline-none focus:border-fuchsia-500 transition-all"
                            />
                        </div>

                        {/* ── ML Model block ── */}
                        {selectedBlock.type === 'ml_model' && (
                            <div className="space-y-1.5">
                                <label className="text-[10px] font-medium text-slate-500 uppercase">Trained Model</label>
                                <select
                                    value={selectedBlock.params.model_id || ''}
                                    onChange={(e) => {
                                        const model = models.find(m => m.id === e.target.value);
                                        setBlocks(prev => prev.map(b => b.id === selectedBlock.id ? { ...b, params: { ...b.params, model_id: e.target.value, label: model?.symbol } } : b));
                                    }}
                                    className="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-sm text-slate-200 outline-none focus:border-fuchsia-500 transition-all font-mono"
                                >
                                    {isLoading ? (
                                        <option>Loading models...</option>
                                    ) : models.length === 0 ? (
                                        <option>No models (train first)</option>
                                    ) : (
                                        <>
                                            <option value="">Select model...</option>
                                            {models.map(m => (
                                                <option key={m.id} value={m.id}>
                                                    {m.symbol} – {m.type} ({(m.accuracy * 100).toFixed(1)}%)
                                                </option>
                                            ))}
                                        </>
                                    )}
                                </select>
                            </div>
                        )}

                        {/* ── Indicator block ── */}
                        {selectedBlock.type === 'indicator' && (
                            <div className="space-y-4">
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-medium text-slate-500 uppercase">Period</label>
                                    <input
                                        type="number"
                                        value={selectedBlock.params.period || 14}
                                        onChange={(e) => setBlocks(prev => prev.map(b => b.id === selectedBlock.id ? { ...b, params: { ...b.params, period: parseInt(e.target.value) || 14 } } : b))}
                                        className="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-sm text-white outline-none focus:border-fuchsia-500"
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-medium text-slate-500 uppercase">Condition</label>
                                    <div className="grid grid-cols-2 gap-1.5">
                                        {['>', '<', 'cross_above', 'cross_below'].map(op => (
                                            <button
                                                key={op}
                                                onClick={() => setBlocks(prev => prev.map(b => b.id === selectedBlock.id ? { ...b, params: { ...b.params, op } } : b))}
                                                className={`p-2 text-[10px] font-bold rounded-lg border transition-all ${selectedBlock.params.op === op ? 'bg-fuchsia-500/20 border-fuchsia-500 text-fuchsia-400' : 'bg-slate-950 border-slate-800 text-slate-500 hover:bg-slate-900'}`}
                                            >
                                                {op.replace('_', ' ').toUpperCase()}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                {/* Threshold (for RSI) */}
                                {(selectedBlock.params.type || '').toLowerCase() === 'rsi' && (
                                    <div className="space-y-1.5">
                                        <label className="text-[10px] font-medium text-slate-500 uppercase">Threshold</label>
                                        <input
                                            type="number"
                                            value={selectedBlock.params.val || 50}
                                            onChange={(e) => setBlocks(prev => prev.map(b => b.id === selectedBlock.id ? { ...b, params: { ...b.params, val: parseFloat(e.target.value) || 50 } } : b))}
                                            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-sm text-white outline-none focus:border-fuchsia-500"
                                            min={0}
                                            max={100}
                                        />
                                    </div>
                                )}
                            </div>
                        )}

                        {/* ── Logic block ── */}
                        {selectedBlock.type === 'logic' && (
                            <div className="space-y-3">
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-medium text-slate-500 uppercase">Gate Type</label>
                                    <div className="grid grid-cols-2 gap-1.5">
                                        {['AND', 'OR'].map(op => (
                                            <button
                                                key={op}
                                                onClick={() => setBlocks(prev => prev.map(b => b.id === selectedBlock.id ? { ...b, params: { ...b.params, op }, label: `${op} Gate` } : b))}
                                                className={`p-2 text-[10px] font-bold rounded-lg border transition-all ${selectedBlock.params.op === op ? 'bg-fuchsia-500/20 border-fuchsia-500 text-fuchsia-400' : 'bg-slate-950 border-slate-800 text-slate-500 hover:bg-slate-900'}`}
                                            >
                                                {op}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-medium text-slate-500 uppercase">Connected Inputs</label>
                                    {(selectedBlock.params.inputs || []).length > 0 ? (
                                        <div className="space-y-1">
                                            {(selectedBlock.params.inputs || []).map((inputId: string) => {
                                                const inputBlock = blocksById[inputId];
                                                return (
                                                    <div key={inputId} className="flex items-center justify-between p-2 bg-slate-950/50 rounded-lg border border-slate-800">
                                                        <span className="text-[10px] text-slate-300 font-medium">{inputBlock?.label || inputId}</span>
                                                        <button
                                                            onClick={() => removeConnection(inputId, selectedBlock.id)}
                                                            className="text-red-400/60 hover:text-red-400"
                                                        >
                                                            <X size={12} />
                                                        </button>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    ) : (
                                        <p className="text-[10px] text-slate-500 italic p-2 bg-slate-950/30 rounded-lg border border-dashed border-slate-800">
                                            Connect blocks to add inputs
                                        </p>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* ── Risk block ── */}
                        {selectedBlock.type === 'risk' && (
                            <div className="space-y-3">
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-medium text-slate-500 uppercase">Type</label>
                                    <div className="grid grid-cols-2 gap-1.5">
                                        {[{ key: 'stop_loss', label: 'Stop Loss' }, { key: 'take_profit', label: 'Take Profit' }].map(rt => (
                                            <button
                                                key={rt.key}
                                                onClick={() => setBlocks(prev => prev.map(b => b.id === selectedBlock.id ? { ...b, params: { ...b.params, risk_type: rt.key }, label: rt.label } : b))}
                                                className={`p-2 text-[10px] font-bold rounded-lg border transition-all ${selectedBlock.params.risk_type === rt.key ? 'bg-fuchsia-500/20 border-fuchsia-500 text-fuchsia-400' : 'bg-slate-950 border-slate-800 text-slate-500 hover:bg-slate-900'}`}
                                            >
                                                {rt.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-medium text-slate-500 uppercase">Threshold %</label>
                                    <input
                                        type="number"
                                        value={selectedBlock.params.threshold || 5}
                                        onChange={(e) => setBlocks(prev => prev.map(b => b.id === selectedBlock.id ? { ...b, params: { ...b.params, threshold: parseFloat(e.target.value) || 5 } } : b))}
                                        className="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-sm text-white outline-none focus:border-fuchsia-500"
                                        min={0.1}
                                        step={0.5}
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-[10px] font-medium text-slate-500 uppercase">Input Block</label>
                                    <p className={`text-[10px] p-2 rounded-lg border ${selectedBlock.params.input ? 'text-slate-300 bg-slate-950/50 border-slate-800' : 'text-slate-500 italic bg-slate-950/30 border-dashed border-slate-800'}`}>
                                        {selectedBlock.params.input ? (blocksById[selectedBlock.params.input]?.label || selectedBlock.params.input) : 'Not connected'}
                                    </p>
                                </div>
                            </div>
                        )}

                        {/* ── Output block ── */}
                        {selectedBlock.type === 'output' && (
                            <div className="space-y-1.5">
                                <label className="text-[10px] font-medium text-slate-500 uppercase">Input Block</label>
                                <p className={`text-[10px] p-2.5 rounded-lg border ${selectedBlock.params.input ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30 font-medium' : 'text-red-400/60 italic bg-red-500/5 border-red-500/20'}`}>
                                    {selectedBlock.params.input
                                        ? `✓ ${blocksById[selectedBlock.params.input]?.label || selectedBlock.params.input}`
                                        : '⚠ Not connected — wire a block here'}
                                </p>
                            </div>
                        )}

                        {/* Delete button */}
                        {selectedBlock.id !== 'root' && (
                            <button
                                onClick={() => deleteBlock(selectedBlock.id)}
                                className="w-full p-3 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 rounded-xl text-xs font-bold flex items-center justify-center gap-2 transition-all mt-6"
                            >
                                <Trash2 size={14} />
                                Remove Block
                            </button>
                        )}
                    </div>
                ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-center opacity-40">
                        <div className="w-12 h-12 bg-slate-800 rounded-2xl flex items-center justify-center mb-4">
                            <ArrowRight className="text-slate-500" strokeWidth={1} />
                        </div>
                        <p className="text-xs text-slate-500">Select a block to<br />configure properties</p>
                        <p className="text-[10px] text-slate-600 mt-4 max-w-[180px] leading-relaxed">
                            Tip: Click an output port (right dot), then click an input port (left dot) to connect blocks.
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default StrategyBuilder;
