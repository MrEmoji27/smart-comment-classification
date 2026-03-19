import './Footer.css';

export default function Footer({ runtimeLabel, backendStatus }) {
  const statusCopy = backendStatus === 'ok'
    ? 'healthy'
    : backendStatus === 'degraded'
      ? 'degraded'
      : backendStatus === 'offline'
        ? 'offline'
        : 'checking';

  const modelStack = [
    runtimeLabel || 'Sentiment runtime',
    'Toxic-BERT',
    'BART MNLI',
    'GoEmotions',
    'Twitter RoBERTa Irony',
    'VADER',
  ];

  return (
    <footer className="site-footer">
      <span className="site-footer-brand">Smart Comment Classification</span>
      <span className="site-footer-inline">
        <span>Backend {statusCopy}</span>
        <span className="site-footer-dot" aria-hidden="true" />
        <span>{modelStack.join(' · ')}</span>
      </span>
    </footer>
  );
}
