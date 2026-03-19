import { useRef, useState } from 'react';
import { Button, Group, Paper, Text, Textarea } from '@mantine/core';
import { AlertCircle, RotateCcw, Send } from 'lucide-react';
import './TextInput.css';

export default function TextInput({ onClassify, loading, hasResult, onClear }) {
  const [text, setText] = useState('');
  const [error, setError] = useState('');
  const textareaRef = useRef(null);

  const charCount = text.length;
  const maxChars = 8192;

  function getCounterClass() {
    if (charCount >= 8000) return 'counter-red';
    if (charCount >= 7000) return 'counter-amber';
    return '';
  }

  function handleSubmit() {
    if (!text.trim()) {
      setError('Please enter a comment.');
      return;
    }

    setError('');
    onClassify(text.trim());
  }

  function handleClear() {
    setText('');
    setError('');
    onClear();

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.focus();
    }
  }

  function handleTextChange(event) {
    setText(event.target.value);
    if (error) setError('');
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      handleSubmit();
    }
  }

  return (
    <Paper className="text-input-panel glass-card animate-fade-in" p={32} radius="lg">
      <h2 className="panel-title">Classify a Comment</h2>
      <p className="panel-desc">
        Enter a comment below and let AI analyze its sentiment, type, and toxicity.
      </p>

      <div className="textarea-wrapper">
        <Textarea
          ref={textareaRef}
          id="comment-textarea"
          className="mantine-textarea"
          placeholder="Enter your comment here..."
          value={text}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          minRows={4}
          maxRows={10}
          autosize
          maxLength={maxChars}
          error={error || undefined}
          aria-describedby="char-count"
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

      <Group className="text-input-actions" gap="sm" mt="md">
        <Button
          className="mantine-primary-button"
          onClick={handleSubmit}
          loading={loading}
          leftSection={!loading ? <Send size={16} /> : null}
          id="classify-text-btn"
          radius="md"
          size="md"
          fullWidth
        >
          Classify
        </Button>

        {hasResult && (
          <Button
            variant="default"
            onClick={handleClear}
            id="clear-text-btn"
            leftSection={<RotateCcw size={16} />}
            radius="md"
            size="md"
          >
            Clear
          </Button>
        )}
      </Group>

      <Text className="input-hint" size="xs" c="dimmed" ta="center" mt="sm">
        Press <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to classify
      </Text>
    </Paper>
  );
}
