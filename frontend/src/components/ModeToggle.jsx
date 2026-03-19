import { FileUp, Type } from 'lucide-react';
import './ModeToggle.css';

export default function ModeToggle({ mode, onModeChange }) {
  return (
    <div className="mode-toggle" role="tablist" aria-label="Input mode">
      <div
        className="mode-toggle-slider"
        style={{ transform: mode === 'file' ? 'translateX(100%)' : 'translateX(0)' }}
      />
      <button
        type="button"
        role="tab"
        aria-selected={mode === 'text'}
        className={`mode-toggle-btn ${mode === 'text' ? 'active' : ''}`}
        onClick={() => onModeChange('text')}
        id="toggle-text-mode"
      >
        <Type size={16} />
        <span>Text Input</span>
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === 'file'}
        className={`mode-toggle-btn ${mode === 'file' ? 'active' : ''}`}
        onClick={() => onModeChange('file')}
        id="toggle-file-mode"
      >
        <FileUp size={16} />
        <span>File Upload</span>
      </button>
    </div>
  );
}
