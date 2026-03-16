import { useReducer, useEffect } from 'react';
import axios from 'axios';
import NavBar from './components/NavBar';
import TextInput from './components/TextInput';
import FileUpload from './components/FileUpload';
import SingleResult from './components/SingleResult';
import BatchResults from './components/BatchResults';
import Footer from './components/Footer';
import './App.css';

const API_URL = 'http://localhost:8000';

const EXAMPLE_COMMENTS = [
  "This app is amazing, it solved all my problems!",
  "smh worst update ever, nothing works anymore",
  "How do I export my data to CSV?",
];

const initialState = {
  mode: 'text', // 'text' | 'file'
  loading: false,
  error: null,
  singleResult: null,
  batchResults: null,
  batchFile: null,
  progress: 0,
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_MODE':
      return { ...initialState, mode: action.payload };
    case 'START_LOADING':
      return { ...state, loading: true, error: null };
    case 'ERROR':
      return { ...state, loading: false, error: action.payload };
    case 'CLEAR_ERROR':
      return { ...state, error: null };
    case 'SET_SINGLE_RESULT':
      return { ...state, loading: false, singleResult: action.payload, error: null };
    case 'SET_BATCH_START':
      return { ...state, batchFile: action.payload, batchResults: null };
    case 'SET_BATCH_PROGRESS':
      return { ...state, progress: action.payload };
    case 'SET_BATCH_RESULTS':
      return { ...state, loading: false, batchResults: action.payload, progress: 100 };
    case 'CLEAR_RESULTS':
      return { ...state, singleResult: null, batchResults: null, error: null };
    default:
      return state;
  }
}

function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { mode, loading, error, singleResult, batchResults, batchFile, progress } = state;

  // Clear toast error after 5s
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        dispatch({ type: 'CLEAR_ERROR' });
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  const handleClassifyText = async (text) => {
    dispatch({ type: 'START_LOADING' });
    try {
      const res = await axios.post(`${API_URL}/classify/text`, { text });
      dispatch({ type: 'SET_SINGLE_RESULT', payload: res.data });
    } catch (err) {
      dispatch({ type: 'ERROR', payload: err.response?.data?.detail || 'An error occurred during classification.' });
    }
  };

  const handleClassifyFile = async (file, column) => {
    dispatch({ type: 'START_LOADING' });

    const formData = new FormData();
    formData.append('file', file);
    if (column) formData.append('column', column);

    try {
      const res = await axios.post(`${API_URL}/classify/file`, formData);

      if (res.data.status === 'needs_column') {
        dispatch({ type: 'ERROR', payload: null });
        return res.data;
      }

      dispatch({ type: 'SET_BATCH_START', payload: file });
      const jobId = res.data.job_id;
      pollJobStatus(jobId);
    } catch (err) {
      dispatch({ type: 'ERROR', payload: err.response?.data?.detail || 'An error occurred during file upload.' });
    }
  };

  const pollJobStatus = async (jobId) => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/classify/status/${jobId}`);
        const data = res.data;

        if (data.status === 'processing') {
          const pct = Math.round((data.processed / data.total) * 100) || 0;
          dispatch({ type: 'SET_BATCH_PROGRESS', payload: pct });
        } else if (data.status === 'done') {
          clearInterval(interval);
          dispatch({ type: 'SET_BATCH_RESULTS', payload: data.results });
        } else if (data.status === 'failed') {
          clearInterval(interval);
          dispatch({ type: 'ERROR', payload: 'Background classification job failed.' });
        }
      } catch (err) {
        clearInterval(interval);
        dispatch({ type: 'ERROR', payload: 'Lost connection to server during polling.' });
      }
    }, 1000);
  };

  return (
    <>
      <a href="#main-content" className="skip-nav">Skip to main content</a>
      <NavBar mode={mode} onModeChange={(m) => dispatch({ type: 'SET_MODE', payload: m })} />

      <main id="main-content" className="app-layout">
        <div className="layout-col">
          {mode === 'text' ? (
            <TextInput
              onClassify={handleClassifyText}
              loading={loading}
              hasResult={!!singleResult}
              onClear={() => dispatch({ type: 'CLEAR_RESULTS' })}
            />
          ) : (
            <>
              <FileUpload
                onClassify={handleClassifyFile}
                loading={loading}
              />
              {loading && !batchResults && mode === 'file' && (
                <div className="progress-container glass-card animate-fade-in mt-4" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100} aria-label="File processing progress">
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
          )}
        </div>

        <div className="layout-col">
           {mode === 'text' && singleResult && (
              <SingleResult result={singleResult} />
           )}

           {mode === 'file' && batchResults && (
              <BatchResults results={batchResults} originalFile={batchFile} />
           )}

           {!singleResult && !batchResults && !loading && mode === 'text' && (
             <div className="empty-results-state animate-fade-in" role="region" aria-label="Try an example">
                <div className="empty-state-content">
                  <p className="empty-state-title">Try an example</p>
                  <div className="example-comments">
                    {EXAMPLE_COMMENTS.map((comment, i) => (
                      <button
                        key={i}
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
        <div className="error-toast animate-slide-in" role="alert">
          {error}
        </div>
      )}

      <Footer />
    </>
  );
}

export default App;
