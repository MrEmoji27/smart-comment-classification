import { useState, useRef } from 'react';
import { CloudUpload, X, FileText, AlertCircle, ChevronDown } from 'lucide-react';
import './FileUpload.css';

const ACCEPTED = ['.csv', '.txt', '.xlsx'];
const MAX_SIZE = 10 * 1024 * 1024; // 10MB

export default function FileUpload({ onClassify, loading }) {
    const [file, setFile] = useState(null);
    const [error, setError] = useState('');
    const [dragging, setDragging] = useState(false);
    const [columns, setColumns] = useState([]);
    const [selectedColumn, setSelectedColumn] = useState('');
    const [needsColumn, setNeedsColumn] = useState(false);
    const inputRef = useRef(null);

    const validateFile = (f) => {
        const ext = '.' + f.name.split('.').pop().toLowerCase();
        if (!ACCEPTED.includes(ext)) {
            setError('Only .csv, .txt, .xlsx files are accepted.');
            return false;
        }
        if (f.size > MAX_SIZE) {
            setError('File exceeds maximum size of 10MB.');
            return false;
        }
        return true;
    };

    const handleFile = (f) => {
        setError('');
        setNeedsColumn(false);
        setColumns([]);
        setSelectedColumn('');
        if (validateFile(f)) {
            setFile(f);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files[0];
        if (f) handleFile(f);
    };

    const handleDragOver = (e) => { e.preventDefault(); setDragging(true); };
    const handleDragLeave = () => setDragging(false);

    const handleKeyDown = (e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !file) {
            e.preventDefault();
            inputRef.current?.click();
        }
    };

    const removeFile = () => {
        setFile(null);
        setError('');
        setNeedsColumn(false);
        setColumns([]);
        if (inputRef.current) inputRef.current.value = '';
    };

    const handleClassify = async () => {
        if (!file) return;
        const result = await onClassify(file, selectedColumn);
        if (result && result.status === 'needs_column') {
            setNeedsColumn(true);
            setColumns(result.columns || []);
            setSelectedColumn(result.columns?.[0] || '');
        }
    };

    const formatSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    return (
        <div className="file-upload-panel glass-card animate-fade-in">
            <h2 className="panel-title">Upload a File</h2>
            <p className="panel-desc">Drag & drop or browse to upload a file for bulk classification.</p>

            <div
                className={`drop-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => !file && inputRef.current?.click()}
                onKeyDown={handleKeyDown}
                role={file ? undefined : 'button'}
                tabIndex={file ? undefined : 0}
                aria-label={file ? undefined : 'Choose a file to upload'}
                id="file-drop-zone"
            >
                <input
                    ref={inputRef}
                    type="file"
                    accept=".csv,.txt,.xlsx"
                    onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
                    hidden
                    aria-label="File input"
                />

                {file ? (
                    <div className="file-chip">
                        <FileText size={20} />
                        <div className="file-chip-info">
                            <span className="file-chip-name">{file.name}</span>
                            <span className="file-chip-size">{formatSize(file.size)}</span>
                        </div>
                        <button className="file-chip-remove" onClick={(e) => { e.stopPropagation(); removeFile(); }} aria-label="Remove file">
                            <X size={16} />
                        </button>
                    </div>
                ) : (
                    <div className="drop-zone-content">
                        <div className="drop-zone-icon"><CloudUpload size={36} /></div>
                        <p className="drop-zone-text">Drop your file here, or <span className="drop-zone-browse">browse</span></p>
                        <p className="drop-zone-hint">Supports CSV, TXT, XLSX &middot; Max 10MB</p>
                    </div>
                )}
            </div>

            <div className="format-pills">
                {ACCEPTED.map(ext => (
                    <span key={ext} className="format-pill">{ext}</span>
                ))}
            </div>

            {needsColumn && columns.length > 0 && (
                <div className="column-selector animate-fade-in">
                    <label htmlFor="column-select" className="column-label">
                        <ChevronDown size={14} />
                        Select the text column to classify:
                    </label>
                    <select
                        className="column-select"
                        value={selectedColumn}
                        onChange={(e) => setSelectedColumn(e.target.value)}
                        id="column-select"
                    >
                        {columns.map(col => (
                            <option key={col} value={col}>{col}</option>
                        ))}
                    </select>
                </div>
            )}

            {error && (
                <div className="input-error" role="alert">
                    <AlertCircle size={14} />
                    <span>{error}</span>
                </div>
            )}

            <button
                className="btn-primary"
                onClick={handleClassify}
                disabled={!file || loading}
                id="classify-file-btn"
                style={{ marginTop: 16 }}
                aria-busy={loading}
            >
                {loading ? (
                    <>
                        <span className="spinner" aria-hidden="true" />
                        <span className="sr-only">Processing file...</span>
                    </>
                ) : (
                    <><CloudUpload size={18} /> Classify File</>
                )}
            </button>
        </div>
    );
}
