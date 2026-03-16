import { useEffect, useRef } from 'react';
import { TrendingUp, TrendingDown, Minus, Clock, AlertTriangle, Tag, AlertCircle, Zap } from 'lucide-react';
import './SingleResult.css';

export default function SingleResult({ result }) {
    const cardRef = useRef(null);

    useEffect(() => {
        if (!result || !cardRef.current) return;
        cardRef.current.classList.add('animate-slide-in');
    }, [result]);

    if (!result) return null;

    const label = result.sentiment || result.label;
    const confidence = result.sentiment_confidence || {};
    const latency_ms = result.latency_ms;
    const commentType = result.comment_type || 'Unknown';
    const isToxic = result.is_toxic || false;
    const toxicityScore = result.toxicity || 0;
    const wordAnalysis = result.word_analysis || [];
    const wordCounts = result.word_counts || { total: 0, positive: 0, neutral: 0, negative: 0 };
    const isUncertain = result.is_uncertain || false;
    const isSarcastic = result.is_sarcastic || false;
    const sarcasmScore = result.sarcasm_score || 0;
    const emotions = result.emotions || [];
    const isEnglish = result.is_english !== undefined ? result.is_english : true;
    const multiSentence = result.multi_sentence || null;

    const labelClass = label.toLowerCase();
    const LabelIcon = label === 'Positive' ? TrendingUp : label === 'Negative' ? TrendingDown : Minus;

    const getWordClass = (sentiment) => {
        if (sentiment === 'Positive') return 'highlight-positive';
        if (sentiment === 'Negative') return 'highlight-negative';
        if (sentiment === 'Neutral') return 'highlight-neutral';
        return '';
    };

    const confPos = Math.round((confidence.positive || 0) * 100);
    const confNeu = Math.round((confidence.neutral || 0) * 100);
    const confNeg = Math.round((confidence.negative || 0) * 100);

    const sentimentColorMap = {
        'Positive': 'var(--positive)',
        'Negative': 'var(--negative)',
        'Neutral': 'var(--neutral)',
    };

    return (
        <div ref={cardRef} className="single-result glass-card" role="region" aria-label="Classification result">
            <h3 className="result-heading">Classification Result</h3>

            {/* Flags row: uncertainty, sarcasm, non-English */}
            {(isUncertain || isSarcastic || !isEnglish) && (
                <div className="result-flags">
                    {isUncertain && (
                        <span className="flag-pill flag-uncertain">
                            <AlertCircle size={13} />
                            Low confidence
                        </span>
                    )}
                    {isSarcastic && (
                        <span className="flag-pill flag-sarcasm">
                            <Zap size={13} />
                            Sarcasm detected ({Math.round(sarcasmScore * 100)}%)
                        </span>
                    )}
                    {!isEnglish && (
                        <span className="flag-pill flag-lang">
                            <AlertCircle size={13} />
                            Non-English text
                        </span>
                    )}
                </div>
            )}

            <div className="result-main">
                <span className={`result-label-pill label-pill ${labelClass}`}>
                    <LabelIcon size={16} />
                    {label}
                </span>

                <span className="result-label-pill label-pill type-pill" style={{ background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border-default)' }}>
                    <Tag size={16} style={{ color: 'var(--accent-blue)' }}/>
                    {commentType}
                </span>

                {isToxic && (
                    <span className="result-label-pill label-pill toxic-pill" style={{ background: 'rgba(255, 68, 68, 0.1)', color: '#ff4444', border: '1px solid rgba(255, 68, 68, 0.2)' }}>
                        <AlertTriangle size={16} />
                        Toxic
                    </span>
                )}

                <div className="result-latency">
                    <Clock size={13} />
                    <span className="result-latency-value">{latency_ms}ms</span>
                </div>
            </div>

            {/* Emotions */}
            {emotions.length > 0 && (
                <div className="emotions-section">
                    <h4 className="section-label">Detected Emotions</h4>
                    <div className="emotion-tags">
                        {emotions.map((emo, i) => (
                            <span key={i} className="emotion-tag" title={`${(emo.score * 100).toFixed(1)}% confidence`}>
                                {emo.label}
                                <span className="emotion-score">{Math.round(emo.score * 100)}%</span>
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* Multi-sentence breakdown */}
            {multiSentence && multiSentence.is_mixed && (
                <div className="multi-sentence-section">
                    <h4 className="section-label">Per-Sentence Breakdown</h4>
                    <div className="sentence-list">
                        {multiSentence.sentences.map((s, i) => (
                            <div key={i} className="sentence-item">
                                <span
                                    className="sentence-dot"
                                    style={{ background: sentimentColorMap[s.sentiment] || 'var(--text-muted)' }}
                                />
                                <span className="sentence-text">{s.text}</span>
                                <span className="sentence-label" style={{ color: sentimentColorMap[s.sentiment] || 'var(--text-muted)' }}>
                                    {s.sentiment}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div className="word-analysis-section">
                <h4 className="confidence-title">Lexical Sentiment Breakdown</h4>

                {wordAnalysis.length > 0 ? (
                    <>
                        <div className="interactive-text-box">
                            {wordAnalysis.map((wordObj, i) => {
                                if (wordObj.sentiment === 'Whitespace') {
                                    return <span key={i}>{wordObj.text}</span>;
                                }
                                return (
                                    <span
                                        key={i}
                                        className={`word-highlight ${getWordClass(wordObj.sentiment)}`}
                                        title={`${wordObj.text} - ${wordObj.sentiment}`}
                                        aria-label={`${wordObj.text}: ${wordObj.sentiment}`}
                                    >
                                        {wordObj.text}
                                    </span>
                                );
                            })}
                        </div>

                        <div className="word-summary-stats">
                            <div className="stat-pill">
                                <strong>{wordCounts.total}</strong> Words
                            </div>
                            <div className="stat-pill pos">
                                <strong>{wordCounts.positive}</strong> Positive
                            </div>
                            <div className="stat-pill neu">
                                <strong>{wordCounts.neutral}</strong> Neutral
                            </div>
                            <div className="stat-pill neg">
                                <strong>{wordCounts.negative}</strong> Negative
                            </div>
                        </div>

                        {isToxic && (
                           <div className="toxicity-bar-container" style={{ marginTop: '20px' }}>
                               <div className="confidence-row-header" style={{ marginBottom: '8px' }}>
                                   <span className="confidence-label">Abuse/Toxicity Detected</span>
                                   <span className="confidence-pct" style={{ color: '#ff4444' }}>
                                       {(toxicityScore * 100).toFixed(1)}%
                                   </span>
                               </div>
                               <div className="confidence-bar-track" style={{ background: 'rgba(255,68,68,0.1)' }}>
                                   <div
                                       className="confidence-bar-fill"
                                       style={{ background: '#ff4444', width: `${Math.min((toxicityScore * 100), 100).toFixed(1)}%` }}
                                   />
                               </div>
                           </div>
                        )}

                        {/* Subtle confidence split — three bars side by side */}
                        <div className="confidence-trio" aria-label={`Confidence: ${confPos}% positive, ${confNeu}% neutral, ${confNeg}% negative`}>
                            <div className="conf-bar-item">
                                <div className="conf-bar-track">
                                    <div className="conf-bar-fill conf-fill-pos" style={{ width: `${confPos}%` }} />
                                </div>
                                <span className="conf-bar-label">{confPos}%</span>
                            </div>
                            <div className="conf-bar-item">
                                <div className="conf-bar-track">
                                    <div className="conf-bar-fill conf-fill-neu" style={{ width: `${confNeu}%` }} />
                                </div>
                                <span className="conf-bar-label">{confNeu}%</span>
                            </div>
                            <div className="conf-bar-item">
                                <div className="conf-bar-track">
                                    <div className="conf-bar-fill conf-fill-neg" style={{ width: `${confNeg}%` }} />
                                </div>
                                <span className="conf-bar-label">{confNeg}%</span>
                            </div>
                        </div>
                    </>
                ) : (
                    <div className="empty-state" style={{ padding: '20px', textAlign: 'center' }}>
                        Word-level analysis not available for this comment.
                    </div>
                )}
            </div>
        </div>
    );
}
