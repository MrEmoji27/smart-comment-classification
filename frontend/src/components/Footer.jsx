import { useState, useEffect } from 'react';
import { Zap } from 'lucide-react';

const ITEMS = [
  'Handles slang, emojis, and informal text',
  'Sentiment · cardiffnlp/twitter-roberta-base-sentiment-latest',
  'Toxicity · unitary/toxic-bert',
  'Type · facebook/bart-large-mnli',
  'Emotions · SamLowe/roberta-base-go_emotions',
  'Sarcasm · cardiffnlp/twitter-roberta-base-irony',
  'Word-level · VADER lexicon',
];

export default function Footer() {
  const [index, setIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setVisible(false);
      setTimeout(() => {
        setIndex(i => (i + 1) % ITEMS.length);
        setVisible(true);
      }, 300);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <footer style={{
      textAlign: 'center',
      padding: '24px',
      color: 'var(--text-secondary)',
      fontSize: '13px',
      marginTop: 'auto',
    }}>
      <div style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '6px 16px',
        background: 'rgba(255,255,255,0.03)',
        borderRadius: 'var(--radius-full)',
        border: '1px solid var(--border-default)',
        color: 'var(--text-secondary)',
        minWidth: '360px',
        justifyContent: 'center',
      }}>
        <Zap size={14} color="var(--accent-blue)" style={{ flexShrink: 0 }} />
        <span style={{
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(4px)',
          transition: 'opacity 300ms ease, transform 300ms ease',
          fontFamily: index > 0 ? 'var(--font-mono)' : 'var(--font-primary)',
          fontSize: index > 0 ? '12px' : '13px',
          whiteSpace: 'nowrap',
        }}>
          {ITEMS[index]}
        </span>
      </div>
    </footer>
  );
}
