import { ActionIcon, Badge, Group, SegmentedControl } from '@mantine/core';
import { Brain, MoonStar, SunMedium } from 'lucide-react';
import './NavBar.css';

export default function NavBar({ mode, onModeChange, theme, onThemeToggle, runtimeLabel, backendStatus }) {
  const statusLabel = backendStatus === 'ok'
    ? 'Online'
    : backendStatus === 'degraded'
      ? 'Degraded'
      : backendStatus === 'offline'
        ? 'Offline'
        : 'Checking';

  const badgeColor = backendStatus === 'ok'
    ? 'green'
    : backendStatus === 'degraded'
      ? 'yellow'
      : 'gray';

  return (
    <nav className="navbar" aria-label="Main navigation">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <div className="navbar-logo-icon">
            <Brain size={24} />
          </div>
          <div>
            <h1 className="navbar-title">Smart Comment Classification</h1>
            <div className="navbar-subtitle-row">
              <p className="navbar-subtitle">Sentiment, type & toxicity analysis</p>
              <Badge className="navbar-status-badge" color={badgeColor} variant="light" radius="xl" size="sm">
                {statusLabel}
              </Badge>
            </div>
          </div>
        </div>

        <Group className="navbar-actions" gap="sm">
          <span className="navbar-runtime" title={runtimeLabel}>{runtimeLabel}</span>

          <SegmentedControl
            value={mode}
            onChange={onModeChange}
            radius="xl"
            size="sm"
            className="navbar-segmented"
            data={[
              { label: 'Text Input', value: 'text' },
              { label: 'File Upload', value: 'file' },
            ]}
          />

          <ActionIcon
            variant="default"
            radius="md"
            size="lg"
            className="theme-toggle"
            onClick={onThemeToggle}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? <SunMedium size={16} /> : <MoonStar size={16} />}
          </ActionIcon>
        </Group>
      </div>
    </nav>
  );
}
