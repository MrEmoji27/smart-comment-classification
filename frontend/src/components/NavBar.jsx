import { Brain } from 'lucide-react';
import ModeToggle from './ModeToggle';
import './NavBar.css';

export default function NavBar({ mode, onModeChange }) {
    return (
        <nav className="navbar" aria-label="Main navigation">
            <div className="navbar-inner">
                <div className="navbar-brand">
                    <div className="navbar-logo-icon">
                        <Brain size={24} />
                    </div>
                    <div>
                        <h1 className="navbar-title">Smart Comment Classification</h1>
                        <p className="navbar-subtitle">Sentiment, type & toxicity analysis</p>
                    </div>
                </div>

                <ModeToggle mode={mode} onModeChange={onModeChange} />
            </div>
        </nav>
    );
}
