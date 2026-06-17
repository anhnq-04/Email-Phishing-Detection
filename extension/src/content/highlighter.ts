import type { EvidenceItem, Severity } from '@/src/types/analysis';

// ── Visual config — Sunset Gradient (Orange to Red) ─────────────────────────
const STYLE_ID = 'phishguard-highlight-styles';
const TOOLTIP_ID = 'phishguard-tooltip';

const SKIP_TAGS = new Set([
    'SCRIPT',
    'STYLE',
    'NOSCRIPT',
    'IFRAME',
    'INPUT',
    'TEXTAREA',
    'SELECT',
    'HEAD',
]);

// ── Listener tracking for event delegation ───────────────────────────────────
let activeMouseOverListener: ((ev: MouseEvent) => void) | null = null;
let activeMouseOutListener: ((ev: MouseEvent) => void) | null = null;
let lastActiveSpan: HTMLElement | null = null;


// ── Inject stylesheet ───────────────────────────────────────────────────────
function ensureStylesheet(): void {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
        @keyframes phishguard-fadein {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        @keyframes phishguard-pulse {
            0%, 100% {
                background-color: rgba(239, 68, 68, 0.04);
                box-shadow: 0 0 3px rgba(239, 68, 68, 0.08);
                text-decoration-color: rgba(239, 68, 68, 0.4);
            }
            50% {
                background-color: rgba(239, 68, 68, 0.12);
                box-shadow: 0 0 10px rgba(239, 68, 68, 0.28);
                text-decoration-color: rgba(239, 68, 68, 0.85);
            }
        }
        [data-phishguard="true"] {
            animation: phishguard-fadein 0.3s ease-out, phishguard-pulse 3s infinite ease-in-out;
            border-radius: 4px;
            padding: 1px 4px;
            cursor: help;
            position: relative;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            text-decoration: underline dotted rgba(239, 68, 68, 0.4) 2px;
            text-underline-offset: 4px;
            display: inline;
            box-decoration-break: clone;
            -webkit-box-decoration-break: clone;
        }
        [data-phishguard="true"]:hover {
            animation: none;
            background-color: rgba(239, 68, 68, 0.18);
            box-shadow: 0 0 12px rgba(239, 68, 68, 0.45);
            text-decoration-style: dotted;
            text-decoration-color: #ef4444;
            text-decoration-thickness: 2px;
        }
        [data-phishguard="true"] [data-phishguard="true"] {
            border-radius: 3px;
            text-decoration: underline dotted rgba(239, 68, 68, 0.75) 2px;
            background-color: rgba(239, 68, 68, 0.14);
            box-shadow: inset 0 0 4px rgba(239, 68, 68, 0.15);
        }
        [data-phishguard="true"] [data-phishguard="true"]:hover {
            background-color: rgba(239, 68, 68, 0.26) !important;
            box-shadow: 0 0 14px rgba(239, 68, 68, 0.6) !important;
            text-decoration-color: #be123c !important;
        }
        #${TOOLTIP_ID} {
            position: fixed;
            z-index: 2147483647;
            max-width: 340px;
            min-width: 180px;
            background: #FFFDFD;
            color: #450a0a;
            border: 1px solid rgba(239, 68, 68, 0.2);
            border-radius: 8px;
            padding: 10px 14px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 12.5px;
            line-height: 1.55;
            box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.08), 0 8px 10px -6px rgba(239, 68, 68, 0.04);
            pointer-events: none;
            opacity: 0;
            transform: translateY(4px);
            transition: opacity 0.18s ease, transform 0.18s ease;
        }
        #${TOOLTIP_ID}.visible {
            opacity: 1;
            transform: translateY(0);
        }
        .pg-tooltip-reason {
            font-size: 12.5px;
            color: #450a0a;
            margin: 0;
            line-height: 1.55;
        }
        .pg-tooltip-label {
            display: inline-block;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: #be123c;
            background: rgba(239, 68, 68, 0.08);
            padding: 2px 6px;
            border-radius: 3px;
            margin-bottom: 6px;
        }
    `;
    document.head.appendChild(style);
}

// ── Tooltip singleton ────────────────────────────────────────────────────────
function ensureTooltip(): HTMLDivElement {
    let el = document.getElementById(TOOLTIP_ID) as HTMLDivElement | null;
    if (el) return el;
    el = document.createElement('div');
    el.id = TOOLTIP_ID;
    document.body.appendChild(el);
    return el;
}

function showTooltip(target: HTMLElement): void {
    const tooltip = ensureTooltip();
    
    // Find all active highlights from target up to body root
    const items: { title: string; desc: string }[] = [];
    let current: HTMLElement | null = target;
    while (current) {
        if (current.getAttribute('data-phishguard') === 'true') {
            const title = current.getAttribute('data-pg-title') ?? '';
            const desc = current.getAttribute('data-pg-desc') ?? '';
            if (title || desc) {
                items.push({ title, desc });
            }
        }
        current = current.parentElement;
    }

    if (items.length === 0) {
        hideTooltip();
        return;
    }

    tooltip.innerHTML = items.map((item, index) => `
        ${index > 0 ? '<div style="margin: 8px 0; border-top: 1px dashed rgba(239, 68, 68, 0.2);"></div>' : ''}
        <span class="pg-tooltip-label">${escapeHtml(item.title)}</span>
        <p class="pg-tooltip-reason">${escapeHtml(item.desc)}</p>
    `).join('');

    // Position near the target
    const rect = target.getBoundingClientRect();
    const ttWidth = 320; // approximate
    let left = rect.left + rect.width / 2 - ttWidth / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - ttWidth - 8));
    
    // Read the actual offsetHeight of tooltip for precise alignment
    const ttHeight = tooltip.offsetHeight || (items.length * 70);
    let top = rect.bottom + 6;
    if (top + ttHeight > window.innerHeight) {
        top = Math.max(8, rect.top - ttHeight - 8);
    }
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;

    requestAnimationFrame(() => tooltip.classList.add('visible'));
}

function hideTooltip(): void {
    const tooltip = document.getElementById(TOOLTIP_ID);
    if (tooltip) tooltip.classList.remove('visible');
}

function escapeHtml(str: string): string {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ── Word boundary checks ─────────────────────────────────────────────────────
function isWordChar(char: string | undefined): boolean {
    if (!char) return false;
    return /^[\p{L}\p{N}_]$/u.test(char);
}

function checkWordBoundary(content: string, startIdx: number, matchLength: number): boolean {
    const firstChar = content[startIdx];
    const lastChar = content[startIdx + matchLength - 1];

    if (isWordChar(firstChar)) {
        const prevChar = startIdx > 0 ? content[startIdx - 1] : undefined;
        if (isWordChar(prevChar)) return false;
    }

    if (isWordChar(lastChar)) {
        const nextChar = startIdx + matchLength < content.length ? content[startIdx + matchLength] : undefined;
        if (isWordChar(nextChar)) return false;
    }

    return true;
}

function escapeRegExp(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function findValidMatchIndex(content: string, matchText: string): { index: number; length: number } | null {
    const cleanText = matchText.trim();
    if (!cleanText) return null;

    // Convert matchText into a whitespace-flexible regex pattern (handling newlines, tabs, and spaces)
    const pieces = cleanText.split(/\s+/);
    const patternStr = pieces.map(piece => escapeRegExp(piece)).join('\\s+');

    try {
        const regex = new RegExp(patternStr, 'gi');
        let match: RegExpExecArray | null;
        while ((match = regex.exec(content)) !== null) {
            const startIdx = match.index;
            const matchLength = match[0].length;

            if (checkWordBoundary(content, startIdx, matchLength)) {
                return { index: startIdx, length: matchLength };
            }
        }
    } catch (e) {
        // Fallback in case of regex error
    }

    // Fallback: simple indexOf
    const idx = content.toLowerCase().indexOf(cleanText.toLowerCase());
    if (idx !== -1 && checkWordBoundary(content, idx, cleanText.length)) {
        return { index: idx, length: cleanText.length };
    }

    return null;
}

// ── Highlight a single text node range ─────────────────────────────────────────
function highlightRangeInNode(
    textNode: Text,
    localStart: number,
    localEnd: number,
    item: EvidenceItem
): void {
    const content = textNode.textContent ?? '';
    if (localStart >= localEnd || localStart < 0 || localEnd > content.length) return;

    const parent = textNode.parentNode;
    if (!parent) return;

    const span = document.createElement('span');
    span.setAttribute('data-phishguard', 'true');
    span.setAttribute('data-pg-title', item.title);
    span.setAttribute('data-pg-desc', item.description);
    span.textContent = content.slice(localStart, localEnd);

    if (localStart > 0) {
        const before = document.createTextNode(content.slice(0, localStart));
        parent.insertBefore(before, textNode);
    }
    
    parent.insertBefore(span, textNode);
    
    if (localEnd < content.length) {
        const after = document.createTextNode(content.slice(localEnd));
        parent.insertBefore(after, textNode);
    }
    
    parent.removeChild(textNode);
}

interface TextNodeMapping {
    node: Text;
    start: number; // start index in flat text
    end: number;   // end index in flat text
}

function getTextNodesMapping(root: Node, skipTags: Set<string>): { flatText: string; mapping: TextNodeMapping[] } {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
        acceptNode: (node) => {
            const parent = node.parentElement;
            if (!parent) return NodeFilter.FILTER_REJECT;
            if (skipTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
            
            // Reject if inside an already highlighted <a> tag to avoid double highlights
            let currentParent: HTMLElement | null = parent;
            while (currentParent && currentParent !== root) {
                if (currentParent.tagName === 'A' && currentParent.getAttribute('data-phishguard') === 'true') {
                    return NodeFilter.FILTER_REJECT;
                }
                currentParent = currentParent.parentElement;
            }

            const style = window.getComputedStyle(parent);
            if (style.display === 'none' || style.visibility === 'hidden') return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
        }
    });

    const mapping: TextNodeMapping[] = [];
    let flatText = '';
    let currentNode: Node | null;

    while ((currentNode = walker.nextNode())) {
        const textNode = currentNode as Text;
        const text = textNode.textContent ?? '';
        if (text.length === 0) continue;

        const start = flatText.length;
        flatText += text;
        const end = flatText.length;

        mapping.push({
            node: textNode,
            start,
            end
        });
    }

    return { flatText, mapping };
}

function highlightFlatRange(
    mapping: TextNodeMapping[],
    matchStart: number,
    matchEnd: number,
    item: EvidenceItem
): void {
    const targets: { node: Text; localStart: number; localEnd: number }[] = [];
    for (const mapItem of mapping) {
        const overlapStart = Math.max(mapItem.start, matchStart);
        const overlapEnd = Math.min(mapItem.end, matchEnd);

        if (overlapStart < overlapEnd) {
            targets.push({
                node: mapItem.node,
                localStart: overlapStart - mapItem.start,
                localEnd: overlapEnd - mapItem.start
            });
        }
    }

    // Highlight each overlapping segment node
    for (const target of targets) {
        highlightRangeInNode(target.node, target.localStart, target.localEnd, item);
    }
}

function findLastSmsBubble(): Element | null {
    const bubbleSelectors = [
        'mws-message-wrapper .text-msg',
        'mws-message-wrapper .msg-content',
        'mws-message-body-wrapper',
        'mws-text-message-body',
        'mws-message-part-content',
        'div[class*="message-body"]',
        'div[class*="text-msg"]',
        '.message-content',
        '.text-msg'
    ];
    for (const selector of bubbleSelectors) {
        const bubbles = Array.from(document.querySelectorAll(selector));
        if (bubbles.length > 0) {
            return bubbles[bubbles.length - 1];
        }
    }
    return null;
}

// ── Public API ──────────────────────────────────────────────────────────────
export function highlightEvidence(items: EvidenceItem[]): void {
    ensureStylesheet();
    ensureTooltip();

    // Scope to Gmail email body or Google Messages last SMS bubble if available, otherwise fallback to document.body
    let root: Element | Document | null = null;
    if (window.location.hostname.includes('messages.google.com')) {
        root = findLastSmsBubble();
    } else {
        const emailBodies = document.querySelectorAll('.a3s.aiL');
        if (emailBodies.length > 0) {
            root = emailBodies[emailBodies.length - 1];
        }
    }
    if (!root) {
        root = document.body;
    }

    // Set up event delegation for hover tooltips
    if (!activeMouseOverListener) {
        activeMouseOverListener = (event: MouseEvent) => {
            const target = event.target as HTMLElement | null;
            if (!target) return;
            const highlightSpan = target.closest('[data-phishguard="true"]') as HTMLElement | null;
            if (highlightSpan && highlightSpan !== lastActiveSpan) {
                lastActiveSpan = highlightSpan;
                showTooltip(highlightSpan);
            }
        };
        document.addEventListener('mouseover', activeMouseOverListener);
    }

    if (!activeMouseOutListener) {
        activeMouseOutListener = (event: MouseEvent) => {
            const relatedTarget = event.relatedTarget as HTMLElement | null;
            const highlightSpan = relatedTarget ? relatedTarget.closest('[data-phishguard="true"]') as HTMLElement | null : null;
            if (highlightSpan !== lastActiveSpan) {
                lastActiveSpan = highlightSpan;
                if (!highlightSpan) {
                    hideTooltip();
                } else {
                    showTooltip(highlightSpan);
                }
            }
        };
        document.addEventListener('mouseout', activeMouseOutListener);
    }

    // Sort items by length of matched_text descending to ensure longer spans are highlighted first.
    // This allows shorter nested spans to be successfully matched inside them.
    const sortedItems = [...items]
        .filter((item) => !!item.matched_text)
        .sort((a, b) => b.matched_text!.length - a.matched_text!.length);

    for (const item of sortedItems) {
        if (!item.matched_text) continue;

        // Check <a> links with matching href (where matched_text is a URL or domain)
        if (item.category === 'url' || item.matched_text.startsWith('http') || item.matched_text.includes('.') || item.matched_text.includes('/')) {
            const links = (root as Element).querySelectorAll('a');
            links.forEach((a) => {
                const href = a.getAttribute('href');
                if (href) {
                    const cleanHref = href.trim().toLowerCase();
                    const cleanMatch = item.matched_text!.trim().toLowerCase();

                    // Normalize helper: remove trailing slashes or protocols for comparison
                    const normHref = cleanHref.replace(/^https?:\/\//, '').replace(/^www\./, '').replace(/\/$/, '');
                    const normMatch = cleanMatch.replace(/^https?:\/\//, '').replace(/^www\./, '').replace(/\/$/, '');

                    if (normHref.includes(normMatch) || normMatch.includes(normHref)) {
                        if (!a.getAttribute('data-phishguard')) {
                            a.setAttribute('data-phishguard', 'true');
                            a.setAttribute('data-pg-title', item.title);
                            a.setAttribute('data-pg-desc', item.description);
                        }
                    }
                }
            });
        }

        // Map visible text nodes in the DOM recursively, bypassing existing highlighted A tags
        const { flatText, mapping } = getTextNodesMapping(root, SKIP_TAGS);

        const cleanText = item.matched_text!.trim();
        const pieces = cleanText.split(/\s+/);
        const patternStr = pieces.map(piece => escapeRegExp(piece)).join('\\s+');

        const matches: { start: number; end: number }[] = [];
        try {
            const regex = new RegExp(patternStr, 'gi');
            let match: RegExpExecArray | null;
            while ((match = regex.exec(flatText)) !== null) {
                const start = match.index;
                const end = start + match[0].length;
                if (checkWordBoundary(flatText, start, match[0].length)) {
                    matches.push({ start, end });
                }
            }
        } catch (e) {
            // Regex match error fallback
        }

        if (matches.length === 0) {
            const idx = flatText.toLowerCase().indexOf(cleanText.toLowerCase());
            if (idx !== -1 && checkWordBoundary(flatText, idx, cleanText.length)) {
                matches.push({ start: idx, end: idx + cleanText.length });
            }
        }

        // Apply highlights in reverse order of matches to keep character offset indexes in mapping valid
        for (let i = matches.length - 1; i >= 0; i--) {
            const { start, end } = matches[i];
            highlightFlatRange(mapping, start, end, item);
        }
    }
}

export function clearHighlights(): void {
    // Remove listeners & clear tracking
    if (activeMouseOverListener) {
        document.removeEventListener('mouseover', activeMouseOverListener);
        activeMouseOverListener = null;
    }
    if (activeMouseOutListener) {
        document.removeEventListener('mouseout', activeMouseOutListener);
        activeMouseOutListener = null;
    }
    lastActiveSpan = null;

    // Clean up <a> tags with data-phishguard without destroying their element structures
    const links = document.querySelectorAll('a[data-phishguard="true"]');
    links.forEach((a) => {
        a.removeAttribute('data-phishguard');
        a.removeAttribute('data-pg-title');
        a.removeAttribute('data-pg-desc');
    });

    // Recursively remove nested highlight span elements from innermost to outermost
    while (true) {
        const spans = document.querySelectorAll('span[data-phishguard="true"]');
        if (spans.length === 0) break;

        let targetSpan: Element | null = null;
        for (const span of Array.from(spans)) {
            if (!span.querySelector('span[data-phishguard="true"]')) {
                targetSpan = span;
                break;
            }
        }
        if (!targetSpan) {
            targetSpan = spans[0];
        }

        const text = document.createTextNode(targetSpan.textContent ?? '');
        targetSpan.parentNode?.replaceChild(text, targetSpan);
    }

    // Remove tooltip
    const tooltip = document.getElementById(TOOLTIP_ID);
    if (tooltip) tooltip.remove();

    // Remove injected stylesheet
    const style = document.getElementById(STYLE_ID);
    if (style) style.remove();
}
