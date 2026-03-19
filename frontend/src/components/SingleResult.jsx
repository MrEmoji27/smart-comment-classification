import { useEffect, useRef } from 'react';
import anime from 'animejs';
import { AlertCircle, AlertTriangle, Clock, Minus, Tag, TrendingDown, TrendingUp, Zap } from 'lucide-react';
import './SingleResult.css';

export default function SingleResult({ result, runtimeLabel }) {
  const cardRef = useRef(null);

  useEffect(() => {
    if (!result || !cardRef.current) return;

    const root = cardRef.current;
    const stagedNodes = root.querySelectorAll(
      '.result-heading, .result-flags, .result-main, .confidence-title, .interactive-text-box, .word-summary-stats, .toxicity-bar-container, .confidence-trio'
    );
    const bars = root.querySelectorAll('.confidence-bar-fill, .conf-bar-fill');

    anime.remove(stagedNodes);
    anime.remove(bars);

    const timeline = anime.timeline({
      easing: 'easeOutExpo',
    });

    timeline.add({
      targets: root,
      opacity: [0, 1],
      translateY: [14, 0],
      duration: 320,
    }).add({
      targets: stagedNodes,
      opacity: [0, 1],
      translateY: [12, 0],
      duration: 420,
      delay: anime.stagger(55),
    }, '-=120').add({
      targets: bars,
      scaleX: [0, 1],
      duration: 560,
      delay: anime.stagger(45),
      easing: 'easeOutQuart',
    }, '-=240');

    return () => {
      timeline.pause();
      anime.remove(root);
      anime.remove(stagedNodes);
      anime.remove(bars);
    };
  }, [result]);

  if (!result) return null;

  const label = result.sentiment || result.label;
  const confidence = result.sentiment_confidence || {};
  const latencyMs = result.latency_ms;
  const commentType = result.comment_type || 'Unknown';
  const isToxic = result.is_toxic || false;
  const toxicityScore = result.toxicity || 0;
  const wordAnalysis = result.word_analysis || [];
  const wordCounts = result.word_counts || { total: 0, positive: 0, neutral: 0, negative: 0 };
  const isUncertain = result.is_uncertain || false;
  const isSarcastic = result.is_sarcastic || false;
  const sarcasmScore = result.sarcasm_score || 0;
  const isEnglish = result.is_english !== undefined ? result.is_english : true;

  const labelClass = label.toLowerCase();
  const LabelIcon = label === 'Positive' ? TrendingUp : label === 'Negative' ? TrendingDown : Minus;

  const confPos = Math.round((confidence.positive || 0) * 100);
  const confNeu = Math.round((confidence.neutral || 0) * 100);
  const confNeg = Math.round((confidence.negative || 0) * 100);

  function getWordClass(sentiment) {
    if (sentiment === 'Positive') return 'highlight-positive';
    if (sentiment === 'Negative') return 'highlight-negative';
    if (sentiment === 'Neutral') return 'highlight-neutral';
    return '';
  }

  return (
    <div ref={cardRef} className="single-result glass-card" role="region" aria-label="Classification result">
      <h3 className="result-heading">Classification Result</h3>

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
        <div className="result-pill-group">
          <span className={`result-label-pill label-pill ${labelClass}`}>
            <LabelIcon size={16} />
            {label}
          </span>

          <span className="result-label-pill label-pill type-pill">
            <Tag size={16} />
            {commentType}
          </span>

          {isToxic && (
            <span className="result-label-pill label-pill toxic-pill">
              <AlertTriangle size={16} />
              Toxic
            </span>
          )}
        </div>

        <div className="result-meta">
          <div className="result-latency">
            <Clock size={13} />
            <span className="result-latency-value">{latencyMs}ms</span>
          </div>
          <div className="result-runtime-chip" title={runtimeLabel}>
            {runtimeLabel}
          </div>
        </div>
      </div>

      <div className="word-analysis-section">
        <h4 className="confidence-title">Lexical Sentiment Breakdown</h4>

        {wordAnalysis.length > 0 ? (
          <>
            <div className="interactive-text-box">
              {wordAnalysis.map((wordObj, index) => {
                if (wordObj.sentiment === 'Whitespace') {
                  return <span key={index}>{wordObj.text}</span>;
                }

                return (
                  <span
                    key={index}
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
              <div className="toxicity-bar-container">
                <div className="confidence-row-header">
                  <span className="confidence-label">Abuse/Toxicity Detected</span>
                  <span className="confidence-pct">{(toxicityScore * 100).toFixed(1)}%</span>
                </div>
                <div className="confidence-bar-track">
                  <div
                    className="confidence-bar-fill toxicity-fill"
                    style={{ width: `${Math.min(toxicityScore * 100, 100).toFixed(1)}%` }}
                  />
                </div>
              </div>
            )}

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
          <div className="empty-state-note">
            Word-level analysis not available for this comment.
          </div>
        )}
      </div>
    </div>
  );
}
