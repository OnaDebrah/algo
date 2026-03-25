'use client'
import React, { useEffect, useRef } from 'react';
import { EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter } from '@codemirror/view';
import { python } from '@codemirror/lang-python';
import { oneDark } from '@codemirror/theme-one-dark';
import { defaultKeymap, indentWithTab } from '@codemirror/commands';
import { bracketMatching, foldGutter, indentOnInput, syntaxHighlighting, defaultHighlightStyle } from '@codemirror/language';
import { highlightSelectionMatches } from '@codemirror/search';
import { autocompletion, closeBrackets } from '@codemirror/autocomplete';

interface CodeEditorProps {
    value: string;
    onChange: (value: string) => void;
    onRun?: () => void;
    placeholder?: string;
    className?: string;
}

export default function CodeEditor({ value, onChange, onRun, placeholder, className = '' }: CodeEditorProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const viewRef = useRef<EditorView | null>(null);
    const onChangeRef = useRef(onChange);
    const onRunRef = useRef(onRun);

    // Keep refs up to date
    onChangeRef.current = onChange;
    onRunRef.current = onRun;

    useEffect(() => {
        if (!containerRef.current) return;

        const runKeymap = keymap.of([
            {
                key: 'Ctrl-Enter',
                mac: 'Cmd-Enter',
                run: () => {
                    onRunRef.current?.();
                    return true;
                },
            },
        ]);

        const updateListener = EditorView.updateListener.of((update) => {
            if (update.docChanged) {
                onChangeRef.current(update.state.doc.toString());
            }
        });

        const customTheme = EditorView.theme({
            '&': {
                height: '100%',
                fontSize: '13px',
                backgroundColor: '#0f172a',
            },
            '.cm-content': {
                fontFamily: '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
                padding: '8px 0',
                caretColor: '#f59e0b',
            },
            '.cm-gutters': {
                backgroundColor: '#0f172a',
                borderRight: '1px solid #1e293b',
                color: '#475569',
            },
            '.cm-activeLineGutter': {
                backgroundColor: '#1e293b',
                color: '#94a3b8',
            },
            '.cm-activeLine': {
                backgroundColor: '#1e293b40',
            },
            '.cm-cursor': {
                borderLeftColor: '#f59e0b',
            },
            '.cm-selectionBackground': {
                backgroundColor: '#334155 !important',
            },
            '&.cm-focused .cm-selectionBackground': {
                backgroundColor: '#334155 !important',
            },
            '.cm-matchingBracket': {
                backgroundColor: '#334155',
                outline: '1px solid #f59e0b40',
            },
            '.cm-placeholder': {
                color: '#475569',
                fontStyle: 'italic',
            },
            '.cm-foldGutter': {
                color: '#475569',
            },
        });

        const state = EditorState.create({
            doc: value,
            extensions: [
                lineNumbers(),
                highlightActiveLineGutter(),
                highlightActiveLine(),
                bracketMatching(),
                closeBrackets(),
                autocompletion(),
                foldGutter(),
                indentOnInput(),
                highlightSelectionMatches(),
                syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
                python(),
                oneDark,
                customTheme,
                runKeymap,
                keymap.of([...defaultKeymap, indentWithTab]),
                updateListener,
                EditorView.lineWrapping,
                EditorState.tabSize.of(4),
                placeholder ? EditorView.contentAttributes.of({ 'aria-label': placeholder }) : [],
            ],
        });

        const view = new EditorView({
            state,
            parent: containerRef.current,
        });

        viewRef.current = view;

        return () => {
            view.destroy();
            viewRef.current = null;
        };
        // Only create editor once
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Update editor content when value changes externally (e.g. template insertion)
    useEffect(() => {
        const view = viewRef.current;
        if (!view) return;
        const currentContent = view.state.doc.toString();
        if (currentContent !== value) {
            view.dispatch({
                changes: { from: 0, to: currentContent.length, insert: value },
            });
        }
    }, [value]);

    return (
        <div
            ref={containerRef}
            className={`overflow-auto rounded-lg border border-slate-700/50 ${className}`}
            style={{ minHeight: '300px' }}
        />
    );
}
