import { useState, useRef } from 'react';
import { Send, RotateCcw, AlertCircle } from 'lucide-react';
import './TextInput.css';

export default function TextInput({ onClassify, loading, hasResult, onClear }) {
    const [text, setText] = useState('');
    const [error, setError] = useState('');
    const textareaRef = useRef(null);

    const charCount = text.length;
    const maxChars = 8192;

    const getCounterClass = () => {
        if (charCount >= 8000) return 'counter-red';
        if (charCount >= 7000) return 'counter-amber';
        return '';
    };

    const handleSubmit = () => {
        if (!text.trim()) {
            setError('Please enter a comment.');
            return;
        }
        setError('');
        onClassify(text.trim());
    };

    const handleClear = () => {
        setText('');
        setError('');
        onClear();
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.focus();
        }
    };

    const handleTextChange = (e) => {
        setText(e.target.value);
        if (error) setError('');

        const ta = e.target;
        ta.style.height = 'auto';
        ta.style.height = Math.min(ta.scrollHeight, 240) + 'px';
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            handleSubmit();
        }
    };

    return (
        <div className="text-input-panel glass-card animate-fade-in">
            <h2 className="panel-title">Classify a Comment</h2>
            <p className="panel-desc">Enter a comment below and let AI analyze its sentiment, type, and toxicity.</p>

            <div className="textarea-wrapper">
                <label htmlFor="comment-textarea" className="sr-only">Comment text</label>
                <textarea
                    ref={textareaRef}
                    id="comment-textarea"
                    className={`text-area ${error ? 'has-error' : ''}`}
                    placeholder="Enter your comment here..."
                    value={text}
                    onChange={handleTextChange}
                    onKeyDown={handleKeyDown}
                    rows={3}
                    maxLength={maxChars}
                    aria-describedby="char-count"
                    aria-invalid={!!error}
                />
                <span id="char-count" className={`char-counter ${getCounterClass()}`}>
                    {charCount.toLocaleString()} / {maxChars.toLocaleString()}
                </span>
            </div>

            {error && (
                <div className="input-error" role="alert">
                    <AlertCircle size={14} />
                    <span>{error}</span>
                </div>
            )}

            <div className="text-input-actions">
                <button
                    className="btn-primary"
                    onClick={handleSubmit}
                    disabled={loading}
                    id="classify-text-btn"
                    aria-busy={loading}
                >
                    {loading ? (
                        <>
                            <span className="spinner" aria-hidden="true" />
                            <span className="sr-only">Classifying...</span>
                        </>
                    ) : (
                        <>
                            <Send size={16} />
                            Classify
                        </>
                    )}
                </button>

                {hasResult && (
                    <button className="btn-secondary" onClick={handleClear} id="clear-text-btn">
                        <RotateCcw size={16} />
                        Clear
                    </button>
                )}
            </div>

            <p className="input-hint">Press <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to classify</p>
        </div>
    );
}
