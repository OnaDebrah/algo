'use client';

import React, { useState } from 'react';
import { MessageSquare, Reply, Send } from 'lucide-react';

interface Comment {
    id: number;
    user_id: number;
    username: string;
    target_type: string;
    target_id: number;
    content: string;
    parent_comment_id: number | null;
    created_at: string | null;
}

interface CommentThreadProps {
    comments: Comment[];
    onAddComment: (content: string, parentId?: number) => Promise<void>;
}

const CommentThread: React.FC<CommentThreadProps> = ({ comments, onAddComment }) => {
    const [newComment, setNewComment] = useState('');
    const [replyTo, setReplyTo] = useState<number | null>(null);
    const [replyText, setReplyText] = useState('');
    const [submitting, setSubmitting] = useState(false);

    const topLevel = comments.filter(c => !c.parent_comment_id);
    const replies = (parentId: number) => comments.filter(c => c.parent_comment_id === parentId);

    const handleSubmit = async () => {
        if (!newComment.trim()) return;
        setSubmitting(true);
        try {
            await onAddComment(newComment.trim());
            setNewComment('');
        } finally {
            setSubmitting(false);
        }
    };

    const handleReply = async (parentId: number) => {
        if (!replyText.trim()) return;
        setSubmitting(true);
        try {
            await onAddComment(replyText.trim(), parentId);
            setReplyText('');
            setReplyTo(null);
        } finally {
            setSubmitting(false);
        }
    };

    const formatDate = (iso: string | null) => {
        if (!iso) return '';
        return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    };

    const renderComment = (comment: Comment, depth: number = 0) => (
        <div key={comment.id} className={`${depth > 0 ? 'ml-8 border-l-2 border-border pl-4' : ''}`}>
            <div className="py-3">
                <div className="flex items-center gap-2 mb-1">
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white text-xs font-bold">
                        {comment.username.charAt(0).toUpperCase()}
                    </div>
                    <span className="text-sm font-medium text-foreground">{comment.username}</span>
                    <span className="text-xs text-muted-foreground">{formatDate(comment.created_at)}</span>
                </div>
                <p className="text-sm text-foreground/80 ml-8">{comment.content}</p>
                <button
                    onClick={() => setReplyTo(replyTo === comment.id ? null : comment.id)}
                    className="ml-8 mt-1 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                    <Reply size={12} /> Reply
                </button>

                {replyTo === comment.id && (
                    <div className="ml-8 mt-2 flex gap-2">
                        <input
                            value={replyText}
                            onChange={e => setReplyText(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && handleReply(comment.id)}
                            placeholder="Write a reply..."
                            className="flex-1 bg-background border border-border rounded-lg px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-violet-500"
                        />
                        <button
                            onClick={() => handleReply(comment.id)}
                            disabled={submitting || !replyText.trim()}
                            className="px-3 py-1.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-sm disabled:opacity-50 transition-colors"
                        >
                            <Send size={14} />
                        </button>
                    </div>
                )}
            </div>
            {replies(comment.id).map(r => renderComment(r, depth + 1))}
        </div>
    );

    return (
        <div className="space-y-1">
            {topLevel.length === 0 && (
                <div className="text-center py-6 text-muted-foreground text-sm">
                    <MessageSquare size={20} className="mx-auto mb-2 opacity-50" />
                    No comments yet
                </div>
            )}
            {topLevel.map(c => renderComment(c))}

            <div className="flex gap-2 pt-3 border-t border-border">
                <input
                    value={newComment}
                    onChange={e => setNewComment(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                    placeholder="Add a comment..."
                    className="flex-1 bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-violet-500"
                />
                <button
                    onClick={handleSubmit}
                    disabled={submitting || !newComment.trim()}
                    className="px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-sm font-medium disabled:opacity-50 transition-colors flex items-center gap-2"
                >
                    <Send size={14} /> Send
                </button>
            </div>
        </div>
    );
};

export default CommentThread;
