import React from 'react';
import ReactDOM from 'react-dom/client';
import Popup from '../popup/Popup';
import { extractMessage } from './extractor';
import { highlightEvidence, clearHighlights } from './highlighter';
import type { EvidenceItem } from '@/src/types/analysis';

// ── Auto-detect Gmail email open & log extracted data (debug) ──
if (window.location.hostname.includes('mail.google.com')) {
    let lastLoggedUrl = '';

    const tryExtractAndLog = () => {
        const emailBody = document.querySelector('.a3s.aiL');
        if (!emailBody) return;

        const currentUrl = window.location.href;
        if (currentUrl === lastLoggedUrl) return;
        lastLoggedUrl = currentUrl;

        const payload = extractMessage();
        console.log('%c[PhishGuard AI] Extracted Email Data:', 'color: #00e676; font-weight: bold; font-size: 14px;');
        console.log('%cSender:', 'color: #ff9800; font-weight: bold;', payload.sender);
        console.log('%cSubject:', 'color: #ff9800; font-weight: bold;', payload.subject);
        console.log('%cBody:', 'color: #ff9800; font-weight: bold;', payload.body.slice(0, 500) + (payload.body.length > 500 ? '...' : ''));
        console.log('%cLinks:', 'color: #ff9800; font-weight: bold;', payload.links);
    };

    const observer = new MutationObserver(() => {
        tryExtractAndLog();
    });

    observer.observe(document.body, { childList: true, subtree: true });
    setTimeout(tryExtractAndLog, 1500);
}

function toggleFloatingPanel() {
    const existing = document.getElementById('phishguard-floating-container');
    if (existing) {
        if (typeof (existing as any)._closePanel === 'function') {
            (existing as any)._closePanel();
        } else {
            existing.remove();
        }
        return;
    }

    // Create outer container element
    const container = document.createElement('div');
    container.id = 'phishguard-floating-container';

    // Premium styling (smooth transitions, floating layout, shadow, rounded corners)
    Object.assign(container.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        width: '400px',
        height: '580px',
        zIndex: '999999999',
        borderRadius: '24px',
        boxShadow: '0 16px 48px rgba(0, 0, 0, 0.18), 0 6px 20px rgba(0, 0, 0, 0.1)',
        border: '1px solid rgba(0, 0, 0, 0.05)',
        backgroundColor: '#FFFFFF',
        overflow: 'hidden',
        transition: 'opacity 0.38s cubic-bezier(0.34, 1.56, 0.64, 1), transform 0.38s cubic-bezier(0.34, 1.56, 0.64, 1)',
        opacity: '0',
        transform: 'translateY(35px) scale(0.95)',
    });

    // Attach Shadow DOM for style isolation (prevents host styles from bleeding in and vice-versa)
    const shadowRoot = container.attachShadow({ mode: 'open' });

    // Inner container for the React app
    const rootDiv = document.createElement('div');
    Object.assign(rootDiv.style, {
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderRadius: '24px',
        overflow: 'hidden',
    });
    shadowRoot.appendChild(rootDiv);

    // Font styling & box-sizing reset
    const styleSheet = document.createElement('style');
    styleSheet.textContent = `
        * {
            box-sizing: border-box;
        }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    `;
    shadowRoot.appendChild(styleSheet);

    const root = ReactDOM.createRoot(rootDiv);

    const handleClose = () => {
        container.style.transition = 'opacity 0.22s cubic-bezier(0.16, 1, 0.3, 1), transform 0.22s cubic-bezier(0.16, 1, 0.3, 1)';
        container.style.opacity = '0';
        container.style.transform = 'translateY(25px) scale(0.96)';
        setTimeout(() => {
            root.unmount();
            container.remove();
        }, 220);
    };

    // Save close function on the element so we can invoke it from external triggers
    (container as any)._closePanel = handleClose;

    // Render Popup React component inside Shadow Root
    root.render(React.createElement(Popup, { onClose: handleClose }));

    // Append to document and trigger animation
    document.body.appendChild(container);
    
    // Force a layout reflow to ensure the transition runs
    container.offsetHeight;
    
    container.style.opacity = '1';
    container.style.transform = 'translateY(0) scale(1)';
}

chrome.runtime.onMessage.addListener(
    (
        message: { type: string },
        _sender,
        sendResponse: (response: unknown) => void
    ) => {
        if (message.type === 'EXTRACT_MESSAGE') {
            try {
                const payload = extractMessage();
                sendResponse({ success: true, data: payload });
            } catch (err) {
                sendResponse({ success: false, error: String(err) });
            }
            return true;
        }

        if (message.type === 'TOGGLE_FLOATING_PANEL') {
            toggleFloatingPanel();
            sendResponse({ success: true });
            return true;
        }

        if (message.type === 'HIGHLIGHT_EVIDENCE') {
            try {
                const items = (message as any).items as EvidenceItem[];
                if (items && items.length > 0) {
                    clearHighlights(); // clear old highlights first
                    highlightEvidence(items);
                }
                sendResponse({ success: true });
            } catch (err) {
                sendResponse({ success: false, error: String(err) });
            }
            return true;
        }

        if (message.type === 'CLEAR_HIGHLIGHTS') {
            try {
                clearHighlights();
                sendResponse({ success: true });
            } catch (err) {
                sendResponse({ success: false, error: String(err) });
            }
            return true;
        }

        return false;
    }
);

