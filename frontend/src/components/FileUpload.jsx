import { useRef, useState } from 'react';
import { Badge, Button, Paper, Select, Text } from '@mantine/core';
import { AlertCircle, ChevronDown, CloudUpload, FileText, X } from 'lucide-react';
import './FileUpload.css';

const ACCEPTED = ['.csv', '.txt', '.xlsx'];
const MAX_SIZE = 10 * 1024 * 1024;

export default function FileUpload({ onClassify, loading }) {
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [columns, setColumns] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState('');
  const [needsColumn, setNeedsColumn] = useState(false);
  const inputRef = useRef(null);

  function validateFile(nextFile) {
    const ext = `.${nextFile.name.split('.').pop().toLowerCase()}`;
    if (!ACCEPTED.includes(ext)) {
      setError('Only .csv, .txt, .xlsx files are accepted.');
      return false;
    }
    if (nextFile.size > MAX_SIZE) {
      setError('File exceeds maximum size of 10MB.');
      return false;
    }
    return true;
  }

  function handleFile(nextFile) {
    setError('');
    setNeedsColumn(false);
    setColumns([]);
    setSelectedColumn('');

    if (validateFile(nextFile)) {
      setFile(nextFile);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    const nextFile = event.dataTransfer.files[0];
    if (nextFile) handleFile(nextFile);
  }

  function handleDragOver(event) {
    event.preventDefault();
    setDragging(true);
  }

  function handleDragLeave() {
    setDragging(false);
  }

  function handleKeyDown(event) {
    if ((event.key === 'Enter' || event.key === ' ') && !file) {
      event.preventDefault();
      inputRef.current?.click();
    }
  }

  function removeFile() {
    setFile(null);
    setError('');
    setNeedsColumn(false);
    setColumns([]);
    setSelectedColumn('');

    if (inputRef.current) {
      inputRef.current.value = '';
    }
  }

  async function handleClassify() {
    if (!file) return;

    const result = await onClassify(file, selectedColumn);
    if (result?.status === 'needs_column') {
      setNeedsColumn(true);
      setColumns(result.columns || []);
      setSelectedColumn(result.columns?.[0] || '');
    }
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  return (
    <Paper className="file-upload-panel glass-card animate-fade-in" p={32} radius="lg">
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
          onChange={(event) => event.target.files[0] && handleFile(event.target.files[0])}
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
            <button className="file-chip-remove" onClick={(event) => { event.stopPropagation(); removeFile(); }} aria-label="Remove file">
              <X size={16} />
            </button>
          </div>
        ) : (
          <div className="drop-zone-content">
            <div className="drop-zone-icon"><CloudUpload size={36} /></div>
            <p className="drop-zone-text">Drop your file here, or <span className="drop-zone-browse">browse</span></p>
            <p className="drop-zone-hint">Supports CSV, TXT, XLSX · Max 10MB</p>
          </div>
        )}
      </div>

      <div className="format-pills">
        {ACCEPTED.map((ext) => (
          <Badge key={ext} variant="light" radius="xl" className="format-pill">{ext}</Badge>
        ))}
      </div>

      {needsColumn && columns.length > 0 && (
        <div className="column-selector animate-fade-in">
          <Text className="column-label" component="label" htmlFor="column-select">
            <ChevronDown size={14} />
            Select the text column to classify:
          </Text>
          <Select
            id="column-select"
            className="mantine-select"
            value={selectedColumn}
            onChange={(value) => setSelectedColumn(value || '')}
            data={columns}
            searchable={false}
            allowDeselect={false}
          />
        </div>
      )}

      {error && (
        <div className="input-error" role="alert">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      <Button
        className="mantine-primary-button"
        onClick={handleClassify}
        disabled={!file}
        loading={loading}
        id="classify-file-btn"
        radius="md"
        size="md"
        fullWidth
        mt={16}
        leftSection={!loading ? <CloudUpload size={18} /> : null}
      >
        Classify File
      </Button>
    </Paper>
  );
}
