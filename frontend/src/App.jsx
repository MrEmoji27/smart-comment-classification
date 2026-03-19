import { Suspense, lazy, useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { Alert, MantineProvider, createTheme } from '@mantine/core';
import { AlertCircle } from 'lucide-react';
import NavBar from './components/NavBar';
import TextInput from './components/TextInput';
import SingleResult from './components/SingleResult';
import Footer from './components/Footer';
import './App.css';

const FileUpload = lazy(() => import('./components/FileUpload'));
const BatchResults = lazy(() => import('./components/BatchResults'));

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

const EXAMPLE_COMMENTS = [
  'This app is amazing, it solved all my problems!',
  'smh worst update ever, nothing works anymore',
  'How do I export my data to CSV?',
];

export default function App() {
  const [mode, setMode] = useState('text');
  const [theme, setTheme] = useState(() => localStorage.getItem('scc-theme') || 'dark');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [singleResult, setSingleResult] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [batchFile, setBatchFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [runtimeLabel, setRuntimeLabel] = useState('Checking runtime');
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('scc-theme', theme);
  }, [theme]);

  const mantineTheme = useMemo(() => createTheme({
    fontFamily: 'var(--font-primary)',
    primaryColor: 'blue',
    defaultRadius: 'md',
    colors: {
      blue: ['#e9f2ff', '#d3e5ff', '#b5d2ff', '#8db7ff', '#659cff', '#3c81ff', '#1e6fd9', '#1553ab', '#103f82', '#0b2d5c'],
    },
  }), []);

  useEffect(() => {
    let isMounted = true;

    async function fetchHealth() {
      try {
        const response = await axios.get(`${API_URL}/health`);
        if (!isMounted) return;

        setRuntimeLabel(
          response.data.active_sentiment_display_name ||
          response.data.sentiment_display_name ||
          response.data.display_name ||
          'Runtime online'
        );
        setBackendStatus(response.data.status || 'ok');
      } catch {
        if (!isMounted) return;
        setRuntimeLabel('Runtime unavailable');
        setBackendStatus('offline');
      }
    }

    fetchHealth();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!error) return undefined;

    const timer = window.setTimeout(() => {
      setError('');
    }, 5000);

    return () => window.clearTimeout(timer);
  }, [error]);

  async function handleClassifyText(text) {
    setLoading(true);
    setError('');
    setBatchResults(null);
    setProgress(0);

    try {
      const response = await axios.post(`${API_URL}/classify/text`, { text });
      setSingleResult(response.data);
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'An error occurred during classification.');
    } finally {
      setLoading(false);
    }
  }

  async function handleClassifyFile(file, column) {
    setLoading(true);
    setError('');
    setSingleResult(null);

    const formData = new FormData();
    formData.append('file', file);
    if (column) formData.append('column', column);

    try {
      const response = await axios.post(`${API_URL}/classify/file`, formData);

      if (response.data.status === 'needs_column') {
        setLoading(false);
        return response.data;
      }

      setBatchFile(file);
      setBatchResults(null);
      pollJobStatus(response.data.job_id);
      return response.data;
    } catch (requestError) {
      setLoading(false);
      setError(requestError.response?.data?.detail || 'An error occurred during file upload.');
      return null;
    }
  }

  function pollJobStatus(jobId) {
    const interval = window.setInterval(async () => {
      try {
        const response = await axios.get(`${API_URL}/classify/status/${jobId}`);
        const data = response.data;

        if (data.status === 'processing') {
          const pct = Math.round((data.processed / data.total) * 100) || 0;
          setProgress(pct);
          return;
        }

        window.clearInterval(interval);

        if (data.status === 'done') {
          setBatchResults(data.results);
          setProgress(100);
          setLoading(false);
          return;
        }

        setLoading(false);
        setError('Background classification job failed.');
      } catch {
        window.clearInterval(interval);
        setLoading(false);
        setError('Lost connection to server during polling.');
      }
    }, 1000);
  }

  function handleModeChange(nextMode) {
    setMode(nextMode);
    setError('');
    setProgress(0);
  }

  function handleClear() {
    setSingleResult(null);
    setBatchResults(null);
    setError('');
  }

  function toggleTheme() {
    setTheme((current) => (current === 'dark' ? 'light' : 'dark'));
  }

  return (
    <MantineProvider theme={mantineTheme} forceColorScheme={theme}>
      <a href="#main-content" className="skip-nav">Skip to main content</a>

      <NavBar
        mode={mode}
        onModeChange={handleModeChange}
        theme={theme}
        onThemeToggle={toggleTheme}
        runtimeLabel={runtimeLabel}
        backendStatus={backendStatus}
      />

      <main id="main-content" className="app-layout">
        <div className="layout-col">
          {mode === 'text' ? (
            <TextInput
              onClassify={handleClassifyText}
              loading={loading}
              hasResult={Boolean(singleResult)}
              onClear={handleClear}
            />
          ) : (
            <Suspense fallback={<div className="empty-results-state animate-fade-in"><p className="empty-state-hint">Loading file tools...</p></div>}>
              <>
                <FileUpload onClassify={handleClassifyFile} loading={loading} />
                {loading && !batchResults && (
                  <div
                    className="progress-container glass-card animate-fade-in mt-4"
                    role="progressbar"
                    aria-valuenow={progress}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label="File processing progress"
                  >
                    <div className="progress-text">
                      <span>Processing file...</span>
                      <span>{progress}%</span>
                    </div>
                    <div className="progress-bar-bg">
                      <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
                    </div>
                  </div>
                )}
              </>
            </Suspense>
          )}
        </div>

        <div className="layout-col">
          {mode === 'text' && singleResult && (
            <SingleResult result={singleResult} runtimeLabel={runtimeLabel} />
          )}

          {mode === 'file' && batchResults && (
            <Suspense fallback={<div className="empty-results-state animate-fade-in"><p className="empty-state-hint">Loading batch results...</p></div>}>
              <BatchResults results={batchResults} originalFile={batchFile} />
            </Suspense>
          )}

          {!singleResult && !batchResults && !loading && mode === 'text' && (
            <div className="empty-results-state animate-fade-in" role="region" aria-label="Try an example">
              <div className="empty-state-content">
                <p className="empty-state-title">Try an example</p>
                <div className="example-comments">
                  {EXAMPLE_COMMENTS.map((comment) => (
                    <button
                      key={comment}
                      className="example-btn"
                      onClick={() => handleClassifyText(comment)}
                    >
                      "{comment}"
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {!singleResult && !batchResults && !loading && mode === 'file' && (
            <div className="empty-results-state animate-fade-in">
              <p className="empty-state-hint">Upload a file to see batch results here.</p>
            </div>
          )}
        </div>
      </main>

      {error && (
        <Alert
          className="error-toast animate-slide-in"
          color="red"
          variant="filled"
          icon={<AlertCircle size={16} />}
          role="alert"
        >
          {error}
        </Alert>
      )}

      <Footer runtimeLabel={runtimeLabel} backendStatus={backendStatus} />
    </MantineProvider>
  );
}
