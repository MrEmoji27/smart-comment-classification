import { useRef } from 'react';
import { ActionIcon, Badge, Group, SegmentedControl } from '@mantine/core';
import anime from 'animejs';
import { Brain, MoonStar, SunMedium } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import './NavBar.css';

export default function NavBar({ mode, onModeChange, theme, onThemeToggle, backendStatus }) {
  const themeToggleRef = useRef(null);
  const modeToggleRef = useRef(null);
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

  function handleThemeClick() {
    if (themeToggleRef.current) {
      anime.remove(themeToggleRef.current);
      anime({
        targets: themeToggleRef.current,
        scale: [1, 0.92, 1.04, 1],
        rotate: [0, theme === 'dark' ? 16 : -16, 0],
        duration: 340,
        easing: 'easeOutQuad',
      });
    }

    onThemeToggle();
  }

  function handleModeToggle(nextMode) {
    if (modeToggleRef.current && nextMode !== mode) {
      anime.remove(modeToggleRef.current);
      anime({
        targets: modeToggleRef.current,
        scale: [1, 0.985, 1],
        duration: 260,
        easing: 'easeOutQuad',
      });
    }

    onModeChange(nextMode);
  }

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
          <motion.div
            ref={modeToggleRef}
            className="navbar-segmented-wrap"
            animate={{ y: mode === 'file' ? 1 : 0 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
          >
            <SegmentedControl
              value={mode}
              onChange={handleModeToggle}
              radius="xl"
              size="sm"
              className="navbar-segmented"
              data={[
                { label: 'Text Input', value: 'text' },
                { label: 'File Upload', value: 'file' },
              ]}
            />
          </motion.div>

          <ActionIcon
            ref={themeToggleRef}
            variant="default"
            radius="md"
            size="lg"
            className="theme-toggle"
            onClick={handleThemeClick}
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.span
                key={theme}
                className="theme-toggle-icon"
                initial={{ opacity: 0, rotate: theme === 'dark' ? -50 : 50, scale: 0.7, y: 2 }}
                animate={{ opacity: 1, rotate: 0, scale: 1, y: 0 }}
                exit={{ opacity: 0, rotate: theme === 'dark' ? 50 : -50, scale: 0.7, y: -2 }}
                transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
              >
                {theme === 'dark' ? <SunMedium size={16} /> : <MoonStar size={16} />}
              </motion.span>
            </AnimatePresence>
          </ActionIcon>
        </Group>
      </div>
    </nav>
  );
}
