'use client';

import React, { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import {
  BookOpen,
  ChevronDown,
  Clock,
  Code2,
  FileText,
  HelpCircle,
  History,
  Loader2,
  Play,
  Sparkles,
  Terminal,
  Trash2,
  Upload,
  Variable,
} from 'lucide-react';
import { api } from '@/utils/api';

const CodeEditor = lazy(() => import('./CodeEditor'));

// ── Types ─────────────────────────────────────────────────────────────────

interface Template {
  id: string;
  name: string;
  description: string;
  code: string;
  category: string;
}

interface ExecutionResult {
  output: string;
  error: string | null;
  execution_time_ms: number;
  variables: Record<string, string>;
  plots: string[];
}

interface HistoryEntry {
  code: string;
  output: string;
  error: string | null;
  execution_time_ms: number;
  timestamp: string;
}

type OutputTab = 'output' | 'variables' | 'help' | 'history' | 'ai';

// ── Help content ──────────────────────────────────────────────────────────

const HELP_SECTIONS = [
  {
    title: 'Data Functions',
    items: [
      {
        name: 'fetch_data(symbol, period="1y")',
        desc: 'Fetch historical OHLCV data as a pandas DataFrame. Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max',
      },
      {
        name: 'fetch_quote(symbol)',
        desc: 'Get current quote data (price, change, volume, marketCap) as a dict',
      },
      {
        name: 'fetch_crypto(symbol)',
        desc: 'Fetch crypto data. Use symbols like "BTC-USD", "ETH-USD", or just "BTC"',
      },
      {
        name: 'backtest_strategy(symbol, strategy_fn, period="1y")',
        desc: 'Backtest a trading strategy. strategy_fn(df) should return a Series of signals (1=buy, -1=sell, 0=hold)',
      },
    ],
  },
  {
    title: 'Available Libraries',
    items: [
      { name: 'pandas (pd)', desc: 'Data manipulation and analysis' },
      { name: 'numpy (np)', desc: 'Numerical computing' },
      { name: 'matplotlib (plt)', desc: 'Plotting and visualization — charts render as images in the Output tab' },
      { name: 'datetime', desc: 'Date and time operations' },
      { name: 'math', desc: 'Mathematical functions' },
      { name: 'statistics', desc: 'Statistical functions' },
      { name: 'json', desc: 'JSON encoding/decoding' },
    ],
  },
  {
    title: 'Limits',
    items: [
      { name: 'Timeout', desc: 'Max 60 seconds per execution' },
      { name: 'Rate limit', desc: '10 executions per minute' },
      { name: 'Output', desc: 'Max 50KB of printed output' },
      { name: 'Security', desc: 'No file I/O, no network (except data helpers), no OS access' },
    ],
  },
];

// ── Component ─────────────────────────────────────────────────────────────

const ResearchLab: React.FC = () => {
  // Editor state
  const [code, setCode] = useState<string>(
    '# Welcome to the Research Lab!\n# Use fetch_data(), fetch_quote(), or fetch_crypto() to get started.\n\ndf = fetch_data("AAPL", period="3mo")\nprint(df.tail())\n'
  );
  const [isRunning, setIsRunning] = useState(false);
  const [executionTime, setExecutionTime] = useState<number | null>(null);

  // Output state
  const [output, setOutput] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [plots, setPlots] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<OutputTab>('output');

  // Templates & history
  const [templates, setTemplates] = useState<Template[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showTemplateDropdown, setShowTemplateDropdown] = useState(false);

  // AI Assistant state
  const [aiPrompt, setAiPrompt] = useState('');
  const [aiSuggestion, setAiSuggestion] = useState('');
  const [isAiLoading, setIsAiLoading] = useState(false);

  const dropdownRef = useRef<HTMLDivElement>(null);

  // Load templates on mount
  useEffect(() => {
    api.research.getTemplates().then((data: Template[]) => setTemplates(data)).catch(() => {});
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowTemplateDropdown(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Execute code
  const handleRun = useCallback(async () => {
    if (isRunning || !code.trim()) return;
    setIsRunning(true);
    setError(null);
    setOutput('');
    setVariables({});
    setPlots([]);
    setExecutionTime(null);
    setActiveTab('output');

    try {
      const result: ExecutionResult = await api.research.execute(code);
      setOutput(result.output);
      setError(result.error);
      setVariables(result.variables);
      setPlots(result.plots || []);
      setExecutionTime(result.execution_time_ms);
      if (result.error) {
        setActiveTab('output');
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : (err as { message?: string })?.message || 'Execution failed';
      setError(msg);
    } finally {
      setIsRunning(false);
    }
  }, [code, isRunning]);

  // Load history
  const loadHistory = useCallback(async () => {
    try {
      const data: HistoryEntry[] = await api.research.getHistory();
      setHistory(data);
    } catch {
      // ignore
    }
  }, []);

  // AI suggestion handler
  const handleAiSuggest = useCallback(async () => {
    if (isAiLoading || !aiPrompt.trim()) return;
    setIsAiLoading(true);
    setAiSuggestion('');
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = await (api.research as any).suggest(aiPrompt, code);
      setAiSuggestion(result.suggestion);
    } catch {
      setAiSuggestion('# Failed to get suggestion. Please try again.');
    } finally {
      setIsAiLoading(false);
    }
  }, [aiPrompt, code, isAiLoading]);

  const selectTemplate = (template: Template) => {
    setCode(template.code);
    setShowTemplateDropdown(false);
  };

  const loadHistoryEntry = (entry: HistoryEntry) => {
    setCode(entry.code);
    setOutput(entry.output);
    setError(entry.error);
    setExecutionTime(entry.execution_time_ms);
    setActiveTab('output');
  };

  // Check if there's a successful backtest result for marketplace publish button
  const hasBacktestResult = variables['result'] && !error;

  // Publish to marketplace state
  const [showPublishModal, setShowPublishModal] = useState(false);
  const [publishName, setPublishName] = useState('');
  const [publishDesc, setPublishDesc] = useState('');
  const [publishCategory, setPublishCategory] = useState('momentum');
  const [isPublishing, setIsPublishing] = useState(false);
  const [publishSuccess, setPublishSuccess] = useState(false);

  const handlePublish = useCallback(async () => {
    if (isPublishing || !publishName.trim()) return;
    setIsPublishing(true);
    try {
      await api.marketplace.publish({
        name: publishName,
        description: publishDesc || 'Strategy created in Research Lab',
        category: publishCategory,
        code: code,
        is_public: true,
      } as never);
      setPublishSuccess(true);
      setTimeout(() => {
        setShowPublishModal(false);
        setPublishSuccess(false);
        setPublishName('');
        setPublishDesc('');
      }, 2000);
    } catch {
      setError('Failed to publish strategy. Please try again.');
    } finally {
      setIsPublishing(false);
    }
  }, [isPublishing, publishName, publishDesc, publishCategory, code]);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 text-sm text-violet-400 mb-1">
            <Code2 size={16} />
            <span className="font-semibold tracking-wide">RESEARCH LAB</span>
          </div>
          <h1 className="text-2xl font-bold text-foreground">
            Python Playground <span className="text-muted-foreground font-normal">Interactive Code Environment</span>
          </h1>
        </div>

        <div className="flex items-center gap-3">
          {executionTime !== null && (
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground bg-card border border-border rounded-lg px-3 py-1.5">
              <Clock size={12} />
              <span>{executionTime}ms</span>
            </div>
          )}
        </div>
      </div>

      {/* Main layout */}
      <div className="flex gap-4 h-[calc(100vh-280px)] min-h-[500px]">
        {/* Left panel — Code editor */}
        <div className="w-[60%] flex flex-col border border-border rounded-xl overflow-hidden bg-card">
          {/* Toolbar */}
          <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-card">
            <div className="flex items-center gap-2">
              <button
                onClick={handleRun}
                disabled={isRunning || !code.trim()}
                className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-medium transition-all
                           bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-50 disabled:cursor-not-allowed
                           shadow-lg shadow-emerald-500/20"
              >
                {isRunning ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Play size={14} />
                )}
                {isRunning ? 'Running...' : 'Run'}
              </button>

              <button
                onClick={() => {
                  setCode('');
                  setOutput('');
                  setError(null);
                  setVariables({});
                  setPlots([]);
                  setExecutionTime(null);
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-muted-foreground
                           hover:text-foreground hover:bg-accent transition-colors"
              >
                <Trash2 size={14} />
                Clear
              </button>

              {/* Template dropdown */}
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setShowTemplateDropdown(!showTemplateDropdown)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-muted-foreground
                             hover:text-foreground hover:bg-accent transition-colors"
                >
                  <FileText size={14} />
                  Templates
                  <ChevronDown size={12} />
                </button>

                {showTemplateDropdown && (
                  <div className="absolute top-full left-0 mt-1 w-72 bg-card border border-border rounded-xl shadow-xl z-50 overflow-hidden">
                    {templates.map((t) => (
                      <button
                        key={t.id}
                        onClick={() => selectTemplate(t)}
                        className="w-full text-left px-4 py-3 hover:bg-accent transition-colors border-b border-border last:border-0"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-foreground">{t.name}</span>
                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-400">
                            {t.category}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-0.5">{t.description}</p>
                      </button>
                    ))}
                    {templates.length === 0 && (
                      <div className="px-4 py-3 text-sm text-muted-foreground">Loading templates...</div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <span className="text-[10px] text-muted-foreground">
              Ctrl+Enter to run
            </span>
          </div>

          {/* Editor area with CodeMirror */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <Suspense fallback={<div className="h-full bg-slate-900 flex items-center justify-center text-slate-500">Loading editor...</div>}>
              <CodeEditor
                value={code}
                onChange={setCode}
                onRun={handleRun}
                placeholder="# Write your Python code here..."
                className="h-full"
              />
            </Suspense>
          </div>

          {/* Publish to Marketplace button */}
          {hasBacktestResult && (
            <div className="px-4 py-2 border-t border-border bg-card">
              <button
                onClick={() => setShowPublishModal(true)}
                className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-medium transition-all
                           bg-violet-600 hover:bg-violet-500 text-white shadow-lg shadow-violet-500/20"
              >
                <Upload size={14} />
                Publish to Marketplace
              </button>
            </div>
          )}
        </div>

        {/* Right panel — Output */}
        <div className="w-[40%] flex flex-col border border-border rounded-xl overflow-hidden bg-card">
          {/* Output tabs */}
          <div className="flex items-center border-b border-border bg-card">
            {(
              [
                { id: 'output' as OutputTab, icon: Terminal, label: 'Output' },
                { id: 'variables' as OutputTab, icon: Variable, label: 'Variables' },
                { id: 'history' as OutputTab, icon: History, label: 'History' },
                { id: 'help' as OutputTab, icon: HelpCircle, label: 'Help' },
                { id: 'ai' as OutputTab, icon: Sparkles, label: 'AI Assistant' },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  if (tab.id === 'history') loadHistory();
                }}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-colors border-b-2 ${
                  activeTab === tab.id
                    ? 'text-violet-400 border-violet-400'
                    : 'text-muted-foreground border-transparent hover:text-foreground'
                }`}
              >
                <tab.icon size={14} />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-auto">
            {/* Running indicator */}
            {isRunning && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <Loader2 size={32} className="animate-spin text-violet-400 mx-auto" />
                  <p className="mt-3 text-sm text-muted-foreground">Executing code...</p>
                </div>
              </div>
            )}

            {/* Output tab */}
            {!isRunning && activeTab === 'output' && (
              <div className="h-full">
                {error && (
                  <div className="px-4 py-3 bg-red-500/10 border-b border-red-500/30">
                    <pre className="text-red-400 text-sm font-mono whitespace-pre-wrap break-words">{error}</pre>
                  </div>
                )}
                {plots.length > 0 && (
                  <div className="px-4 py-3 space-y-3 border-b border-border/50">
                    {plots.map((plot, i) => (
                      <div key={i} className="rounded-lg overflow-hidden border border-border/50">
                        <img
                          src={`data:image/png;base64,${plot}`}
                          alt={`Plot ${i + 1}`}
                          className="w-full"
                        />
                      </div>
                    ))}
                  </div>
                )}
                <pre className="px-4 py-4 text-emerald-400 text-sm font-mono whitespace-pre-wrap break-words
                               bg-slate-900/50 dark:bg-slate-950/50 h-full">
                  {output || (
                    <span className="text-slate-600">
                      {/* Show placeholder when no output yet */}
                      Run your code to see output here...
                    </span>
                  )}
                </pre>
              </div>
            )}

            {/* Variables tab */}
            {!isRunning && activeTab === 'variables' && (
              <div className="p-4">
                {Object.keys(variables).length > 0 ? (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-muted-foreground border-b border-border">
                        <th className="pb-2 pr-4 font-medium">Name</th>
                        <th className="pb-2 font-medium">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(variables).map(([name, value]) => (
                        <tr key={name} className="border-b border-border/50">
                          <td className="py-2 pr-4 font-mono text-violet-400">{name}</td>
                          <td className="py-2 font-mono text-foreground text-xs">
                            <pre className="whitespace-pre-wrap break-words max-h-32 overflow-auto">
                              {value}
                            </pre>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="text-muted-foreground text-sm">
                    No variables to display. Run code that assigns variables to see them here.
                  </p>
                )}
              </div>
            )}

            {/* History tab */}
            {!isRunning && activeTab === 'history' && (
              <div className="p-4 space-y-2">
                {history.length > 0 ? (
                  history.map((entry, i) => (
                    <button
                      key={i}
                      onClick={() => loadHistoryEntry(entry)}
                      className="w-full text-left p-3 rounded-lg border border-border hover:bg-accent transition-colors"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-muted-foreground">
                          {new Date(entry.timestamp).toLocaleString()}
                        </span>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          entry.error
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-emerald-500/20 text-emerald-400'
                        }`}>
                          {entry.error ? 'Error' : `${entry.execution_time_ms}ms`}
                        </span>
                      </div>
                      <pre className="text-xs font-mono text-foreground/80 truncate">
                        {entry.code.split('\n').find((l) => l.trim() && !l.trim().startsWith('#')) || entry.code.split('\n')[0]}
                      </pre>
                    </button>
                  ))
                ) : (
                  <p className="text-muted-foreground text-sm">
                    No execution history yet. Run some code to see your history here.
                  </p>
                )}
              </div>
            )}

            {/* Help tab */}
            {!isRunning && activeTab === 'help' && (
              <div className="p-4 space-y-6">
                {HELP_SECTIONS.map((section) => (
                  <div key={section.title}>
                    <h3 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                      <BookOpen size={14} className="text-violet-400" />
                      {section.title}
                    </h3>
                    <div className="space-y-2">
                      {section.items.map((item) => (
                        <div key={item.name} className="pl-4 border-l-2 border-border">
                          <code className="text-xs font-mono text-violet-400">{item.name}</code>
                          <p className="text-xs text-muted-foreground mt-0.5">{item.desc}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* AI Assistant tab */}
            {!isRunning && activeTab === 'ai' && (
              <div className="p-4 space-y-4">
                <div>
                  <label className="text-sm font-medium text-foreground mb-2 block">
                    Describe what you want to build:
                  </label>
                  <textarea
                    value={aiPrompt}
                    onChange={(e) => setAiPrompt(e.target.value)}
                    placeholder="e.g., Create a momentum strategy that buys when RSI is below 30 and sells above 70..."
                    className="w-full h-24 bg-slate-900 dark:bg-slate-950 text-foreground text-sm rounded-lg
                               border border-border p-3 resize-none outline-none focus:border-violet-500
                               placeholder:text-muted-foreground/50"
                  />
                </div>
                <button
                  onClick={handleAiSuggest}
                  disabled={isAiLoading || !aiPrompt.trim()}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all
                             bg-violet-600 hover:bg-violet-500 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isAiLoading ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Sparkles size={14} />
                  )}
                  {isAiLoading ? 'Generating...' : 'Suggest Code'}
                </button>

                {aiSuggestion && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-foreground">Suggested Code:</span>
                      <button
                        onClick={() => setCode(prev => prev + '\n\n' + aiSuggestion)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                                   bg-emerald-600 hover:bg-emerald-500 text-white transition-colors"
                      >
                        <Code2 size={12} />
                        Insert into Editor
                      </button>
                    </div>
                    <pre className="bg-slate-900 dark:bg-slate-950 text-emerald-400 text-sm font-mono p-4 rounded-lg
                                    border border-border/50 whitespace-pre-wrap overflow-auto max-h-64">
                      {aiSuggestion}
                    </pre>
                  </div>
                )}

                {!aiSuggestion && !isAiLoading && (
                  <div className="text-sm text-muted-foreground space-y-2">
                    <p className="font-medium">Try asking:</p>
                    <div className="space-y-1">
                      {[
                        "Fetch AAPL data and plot a candlestick-like chart",
                        "Create an RSI-based trading strategy and backtest it",
                        "Compare the correlation between tech stocks",
                        "Calculate the optimal portfolio weights for FAANG stocks",
                      ].map((suggestion, i) => (
                        <button
                          key={i}
                          onClick={() => setAiPrompt(suggestion)}
                          className="block w-full text-left px-3 py-2 text-xs rounded-lg hover:bg-accent
                                     transition-colors text-violet-400"
                        >
                          &rarr; {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Publish to Marketplace Modal */}
      {showPublishModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-card border border-border rounded-2xl shadow-2xl w-full max-w-md p-6 space-y-4">
            {publishSuccess ? (
              <div className="text-center py-8">
                <div className="w-16 h-16 rounded-full bg-emerald-500/20 flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h3 className="text-lg font-bold text-foreground">Published!</h3>
                <p className="text-sm text-muted-foreground mt-1">Your strategy is now live on the Marketplace.</p>
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-bold text-foreground flex items-center gap-2">
                    <Upload size={18} className="text-violet-400" />
                    Publish to Marketplace
                  </h3>
                  <button
                    onClick={() => setShowPublishModal(false)}
                    className="text-muted-foreground hover:text-foreground text-lg"
                  >
                    ✕
                  </button>
                </div>

                <div className="space-y-3">
                  <div>
                    <label className="text-sm font-medium text-foreground block mb-1">Strategy Name *</label>
                    <input
                      type="text"
                      value={publishName}
                      onChange={(e) => setPublishName(e.target.value)}
                      placeholder="e.g., RSI Momentum Strategy"
                      className="w-full bg-slate-900 dark:bg-slate-950 border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:border-violet-500"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium text-foreground block mb-1">Description</label>
                    <textarea
                      value={publishDesc}
                      onChange={(e) => setPublishDesc(e.target.value)}
                      placeholder="Describe how your strategy works, what signals it uses..."
                      rows={3}
                      className="w-full bg-slate-900 dark:bg-slate-950 border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:border-violet-500 resize-none"
                    />
                  </div>

                  <div>
                    <label className="text-sm font-medium text-foreground block mb-1">Category</label>
                    <select
                      value={publishCategory}
                      onChange={(e) => setPublishCategory(e.target.value)}
                      className="w-full bg-slate-900 dark:bg-slate-950 border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:border-violet-500"
                    >
                      <option value="momentum">Momentum</option>
                      <option value="mean_reversion">Mean Reversion</option>
                      <option value="trend_following">Trend Following</option>
                      <option value="statistical_arbitrage">Statistical Arbitrage</option>
                      <option value="machine_learning">Machine Learning</option>
                      <option value="options">Options</option>
                      <option value="crypto">Crypto</option>
                      <option value="other">Other</option>
                    </select>
                  </div>

                  <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2 text-xs text-amber-400">
                    Your code will be shared publicly. Make sure it doesn&apos;t contain sensitive data.
                  </div>
                </div>

                <div className="flex items-center gap-3 pt-2">
                  <button
                    onClick={() => setShowPublishModal(false)}
                    className="flex-1 px-4 py-2 rounded-lg text-sm font-medium border border-border text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handlePublish}
                    disabled={isPublishing || !publishName.trim()}
                    className="flex-1 flex items-center justify-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium
                               bg-violet-600 hover:bg-violet-500 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    {isPublishing ? (
                      <Loader2 size={14} className="animate-spin" />
                    ) : (
                      <Upload size={14} />
                    )}
                    {isPublishing ? 'Publishing...' : 'Publish Strategy'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResearchLab;
