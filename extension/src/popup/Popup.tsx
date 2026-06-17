'use client';

import { useState, useEffect, useCallback } from 'react';
import type { PopupState, AnalysisResponse, ExtractedMessage } from '@/src/types/analysis';
import { analyzeMessage, checkBackendStatus } from '@/src/api/client';
import { extractMessage } from '@/src/content/extractor';
import { highlightEvidence, clearHighlights } from '@/src/content/highlighter';

// ─── Status config ──────────────────────────────────────────────────────────
const STATUS_CONFIG = {
    phishing: {
        label: 'Nguy hiểm',
        sublabel: 'Tin nhắn này có dấu hiệu lừa đảo.',
        icon: 'danger' as const,
        bg: '#FFF1F0', border: '#FFCCC7', iconBg: '#FF4D4F',
        textColor: '#CF1322', scoreColor: '#CF1322',
        evidenceBg: '#FFF1F0', evidenceBorder: '#FFD8D5',
    },
    suspicious: {
        label: 'Cần xác minh',
        sublabel: 'Hãy kiểm tra kỹ trước khi tiếp tục.',
        icon: 'warning' as const,
        bg: '#FFFBE6', border: '#FFE58F', iconBg: '#FAAD14',
        textColor: '#AD6800', scoreColor: '#AD6800',
        evidenceBg: '#FFFBE6', evidenceBorder: '#FFEEBA',
    },
    legit: {
        label: 'An toàn',
        sublabel: 'Tin nhắn này có vẻ an toàn.',
        icon: 'safe' as const,
        bg: '#F6FFED', border: '#B7EB8F', iconBg: '#52C41A',
        textColor: '#237804', scoreColor: '#237804',
        evidenceBg: '#F6FFED', evidenceBorder: '#D1F0C0',
    },
    unknown: {
        label: 'Không rõ',
        sublabel: 'Chưa thể xác định mức độ an toàn.',
        icon: 'unknown' as const,
        bg: '#F5F5F5', border: '#D9D9D9', iconBg: '#8C8C8C',
        textColor: '#595959', scoreColor: '#595959',
        evidenceBg: '#F5F5F5', evidenceBorder: '#E8E8E8',
    },
} as const;

const SEV_COLORS: Record<string, { dot: string; bg: string; border: string }> = {
    critical: { dot: '#FF4D4F', bg: '#FFF1F0', border: '#FFCCC7' },
    high:     { dot: '#FA8C16', bg: '#FFF7E6', border: '#FFE4B5' },
    medium:   { dot: '#FAAD14', bg: '#FFFBE6', border: '#FFEEBA' },
    low:      { dot: '#8C8C8C', bg: '#FAFAFA', border: '#F0F0F0' },
};

// ─── SVG Icons ───────────────────────────────────────────────────────────────
function ShieldIcon({ size = 22, color = '#434343' }: { size?: number; color?: string }) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
            <path d="M12 2L3 6v6c0 5.25 3.75 10.15 9 11.35C17.25 22.15 21 17.25 21 12V6L12 2z"
                stroke={color} strokeWidth="2" strokeLinejoin="round" />
            <path d="M9 12l2 2 4-4" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    );
}

function StatusIllustration({ type }: { type: 'safe' | 'warning' | 'danger' | 'unknown' }) {
    if (type === 'safe') {
        return (
            <svg width="48" height="48" viewBox="0 0 64 64" fill="none">
                <circle cx="32" cy="32" r="28" fill="#F6FFED" stroke="#B7EB8F" strokeWidth="2" />
                <path d="M22 32l7 7 13-13" stroke="#52C41A" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
        );
    }
    if (type === 'warning') {
        return (
            <svg width="48" height="48" viewBox="0 0 64 64" fill="none">
                <circle cx="32" cy="32" r="28" fill="#FFFBE6" stroke="#FFE58F" strokeWidth="2" />
                <path d="M32 20v16" stroke="#FAAD14" strokeWidth="3" strokeLinecap="round" />
                <circle cx="32" cy="44" r="2" fill="#FAAD14" />
            </svg>
        );
    }
    if (type === 'danger') {
        return (
            <svg width="48" height="48" viewBox="0 0 64 64" fill="none">
                <circle cx="32" cy="32" r="28" fill="#FFF1F0" stroke="#FFCCC7" strokeWidth="2" />
                <path d="M24 24l16 16M40 24l-16 16" stroke="#FF4D4F" strokeWidth="3" strokeLinecap="round" />
            </svg>
        );
    }
    return (
        <svg width="48" height="48" viewBox="0 0 64 64" fill="none">
            <circle cx="32" cy="32" r="28" fill="#F5F5F5" stroke="#D9D9D9" strokeWidth="2" />
            <path d="M29 28c0-1.7 1.3-3 3-3s3 1.3 3 3c0 1.5-1 2.5-2.5 3S31 33 31 34" stroke="#8C8C8C" strokeWidth="2.5" strokeLinecap="round" />
            <circle cx="32" cy="40" r="2" fill="#8C8C8C" />
        </svg>
    );
}

function SpinnerIcon() {
    return (
        <svg viewBox="0 0 24 24" fill="none" style={{ width: 32, height: 32, animation: 'phishguard-spin 1s linear infinite', color: '#595959' }}>
            <style>{`@keyframes phishguard-spin { 100% { transform: rotate(360deg); } }`}</style>
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" style={{ opacity: 0.15 }} />
            <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" style={{ opacity: 0.7 }} />
        </svg>
    );
}

function ChevronIcon({ open }: { open: boolean }) {
    return (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{ transition: 'transform .2s', transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}>
            <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
    );
}

// ─── Chrome messaging helpers ────────────────────────────────────────────────
function sendToContentScript(tabId: number, message: { type: string }): Promise<unknown> {
    return new Promise((resolve) => {
        if (typeof chrome === 'undefined' || !chrome.tabs) {
            resolve({ success: false, error: 'chrome API unavailable' });
            return;
        }
        chrome.tabs.sendMessage(tabId, message, (response) => {
            if (chrome.runtime.lastError) resolve({ success: false, error: chrome.runtime.lastError.message });
            else resolve(response);
        });
    });
}

async function getActiveTabId(): Promise<number | null> {
    if (typeof chrome === 'undefined' || !chrome.tabs) return null;
    return new Promise((resolve) => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            resolve(tabs[0]?.id ?? null);
        });
    });
}

// ─── Evidence list component ─────────────────────────────────────────────────
function EvidenceList({ result }: { result: AnalysisResponse }) {
    const items = result.xai_report?.evidence_items ?? [];
    const evidenceStrings = result.evidence ?? [];

    if (items.length > 0) {
        return (
            <div style={{ padding: '0 16px 4px', display: 'flex', flexDirection: 'column', gap: 5 }}>
                <div style={{ fontSize: 11, color: '#8C8C8C', fontWeight: 600, marginTop: 8, marginBottom: 2 }}>
                    Bằng chứng phát hiện ({items.length})
                </div>
                {items.map((item, i) => {
                    const sev = SEV_COLORS[item.severity] ?? SEV_COLORS.low;
                    return (
                        <div key={i} style={{
                            background: '#FFFFFF', border: '1px solid #EBEBEB',
                            borderRadius: 8, padding: '8px 10px',
                            borderLeft: `3px solid ${sev.dot}`,
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                                <span style={{
                                    width: 7, height: 7, borderRadius: '50%',
                                    background: sev.dot, display: 'inline-block', flexShrink: 0,
                                }} />
                                <span style={{ fontSize: 12, fontWeight: 600, color: '#262626' }}>
                                    {item.title}
                                </span>
                            </div>
                            <p style={{ fontSize: 11, color: '#595959', margin: 0, lineHeight: 1.5 }}>
                                {item.description}
                            </p>
                            {item.matched_text && (
                                <div style={{
                                    fontSize: 10, color: '#8B7000', background: 'rgba(250, 204, 21, 0.15)',
                                    borderRadius: 4, padding: '3px 6px', marginTop: 5,
                                    fontFamily: 'monospace', wordBreak: 'break-all', lineHeight: 1.4,
                                    border: '1px solid rgba(250, 204, 21, 0.3)',
                                }}>
                                    "{item.matched_text}"
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        );
    }

    if (evidenceStrings.length > 0) {
        return (
            <div style={{ padding: '0 16px 4px', display: 'flex', flexDirection: 'column', gap: 5 }}>
                <div style={{ fontSize: 11, color: '#8C8C8C', fontWeight: 600, marginTop: 8 }}>
                    Chi tiết phân tích
                </div>
                {evidenceStrings.map((text, i) => (
                    <div key={i} style={{
                        background: '#FAFAFA', border: '1px solid #F0F0F0',
                        borderRadius: 8, padding: '7px 10px',
                    }}>
                        <p style={{ fontSize: 11, color: '#595959', margin: 0, lineHeight: 1.5 }}>
                            {text}
                        </p>
                    </div>
                ))}
            </div>
        );
    }

    return null;
}

// ─── Recommendation box ──────────────────────────────────────────────────────
function RecommendationBox({ result }: { result: AnalysisResponse }) {
    const rec = result.xai_report?.recommendation;
    if (!rec) return null;

    return (
        <div style={{
            margin: '6px 16px 0', padding: '8px 10px', borderRadius: 8,
            background: '#F9F9F9', border: '1px solid #EBEBEB',
            display: 'flex', gap: 8, alignItems: 'flex-start',
        }}>
            <span style={{ fontSize: 13, flexShrink: 0 }}>💡</span>
            <p style={{ margin: 0, fontSize: 11, color: '#595959', lineHeight: 1.5 }}>{rec}</p>
        </div>
    );
}

// ─── Technical details (collapsed) ───────────────────────────────────────────
function TechnicalDetails({ result }: { result: AnalysisResponse }) {
    const [open, setOpen] = useState(false);

    return (
        <div style={{ borderTop: '1px solid #F0F0F0', marginTop: 4 }}>
            <button
                onClick={() => setOpen((v) => !v)}
                style={{
                    width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '8px 16px', background: 'none', border: 'none', cursor: 'pointer',
                    color: '#8C8C8C', fontSize: 11, fontFamily: 'inherit',
                }}
            >
                <span>Thông tin kỹ thuật</span>
                <ChevronIcon open={open} />
            </button>

            {open && (
                <div style={{ padding: '0 16px 12px', display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        <TechPill label={`Rủi ro: ${Math.round(result.risk_score * 100)}%`} />
                        <TechPill label={`Tin cậy: ${Math.round(result.confidence * 100)}%`} />
                        <TechPill label={`Mức: ${result.risk_level}`} />
                        <TechPill label={`Hành động: ${result.action}`} />
                    </div>

                    {result.signals?.length > 0 && (
                        <div>
                            <div style={{ fontSize: 10, color: '#BFBFBF', marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Tín hiệu</div>
                            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                                {result.signals.map((s) => (
                                    <span key={s} style={{ fontSize: 10, background: '#F5F5F5', color: '#595959', padding: '2px 7px', borderRadius: 4, fontFamily: 'monospace' }}>{s}</span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

function TechPill({ label }: { label: string }) {
    return (
        <span style={{ fontSize: 10, background: '#F5F5F5', border: '1px solid #E8E8E8', color: '#595959', padding: '2px 7px', borderRadius: 20 }}>
            {label}
        </span>
    );
}

// ─── Main component ──────────────────────────────────────────────────────────
interface PopupProps {
    onClose?: () => void;
}

export default function Popup({ onClose }: PopupProps) {
    const [state, setState] = useState<PopupState>('idle');
    const [result, setResult] = useState<AnalysisResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [usedMock, setUsedMock] = useState(false);
    const [backendOnline, setBackendOnline] = useState(false);
    const [tabId, setTabId] = useState<number | null>(null);

    // Phục hồi trạng thái cũ khi mount
    useEffect(() => {
        checkBackendStatus().then(setBackendOnline);
        getActiveTabId().then(setTabId);

        if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
            chrome.storage.local.get([
                'phishguard_popup_state',
                'phishguard_popup_result',
                'phishguard_popup_error',
                'phishguard_popup_mock'
            ], (res) => {
                if (res.phishguard_popup_state) {
                    setState(res.phishguard_popup_state);
                }
                if (res.phishguard_popup_result) {
                    setResult(res.phishguard_popup_result);
                }
                if (res.phishguard_popup_error) {
                    setError(res.phishguard_popup_error);
                }
                if (res.phishguard_popup_mock !== undefined) {
                    setUsedMock(res.phishguard_popup_mock);
                }
            });
        }
    }, []);

    // Tự động lưu trạng thái khi thay đổi
    useEffect(() => {
        if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
            chrome.storage.local.set({
                phishguard_popup_state: state,
                phishguard_popup_result: result,
                phishguard_popup_error: error,
                phishguard_popup_mock: usedMock
            });
        }
    }, [state, result, error, usedMock]);

    // Gửi evidence items đến content script để highlight trên email
    const triggerHighlight = useCallback((data: AnalysisResponse) => {
        const items = data.xai_report?.evidence_items ?? [];
        if (items.length === 0) return;

        const isFloatingPanel = typeof chrome === 'undefined' || !chrome.tabs || !chrome.tabs.sendMessage;
        if (isFloatingPanel) {
            // Đang chạy trực tiếp trong trang — gọi highlighter trực tiếp
            clearHighlights();
            highlightEvidence(items);
        } else if (tabId !== null) {
            // Popup mode — gửi message sang content script
            sendToContentScript(tabId, { type: 'HIGHLIGHT_EVIDENCE', items } as any);
        }
    }, [tabId]);

    // Xoá highlight khi reset
    const triggerClearHighlight = useCallback(() => {
        const isFloatingPanel = typeof chrome === 'undefined' || !chrome.tabs || !chrome.tabs.sendMessage;
        if (isFloatingPanel) {
            clearHighlights();
        } else if (tabId !== null) {
            sendToContentScript(tabId, { type: 'CLEAR_HIGHLIGHTS' } as any);
        }
    }, [tabId]);

    const handleDetect = useCallback(async () => {
        setState('loading');
        setError(null);
        setResult(null);

        try {
            let extracted: ExtractedMessage = { sender: '', subject: '', body: '', links: [] };
            
            // Nếu đang chạy dưới dạng Content Script Panel (không có chrome.tabs hoặc query)
            if (typeof chrome === 'undefined' || !chrome.tabs || !chrome.tabs.sendMessage) {
                extracted = extractMessage();
            } else if (tabId !== null) {
                const resp = (await sendToContentScript(tabId, { type: 'EXTRACT_MESSAGE' })) as { success: boolean; data?: ExtractedMessage };
                if (resp?.success && resp.data) extracted = resp.data;
            }

            if (!extracted || !extracted.body?.trim() || (extracted.message_type === 'sms' && !extracted.sender?.trim())) {
                const errorMsg = extracted?.message_type === 'sms'
                    ? 'Không tìm thấy tin nhắn SMS cần xác định. Vui lòng chọn một cuộc trò chuyện để kiểm tra.'
                    : 'Không tìm thấy email cần xác định. Vui lòng mở hoặc chọn một email để kiểm tra.';
                throw new Error(errorMsg);
            }

            const payload = {
                sender_email: extracted.sender,
                subject: extracted.subject,
                body: extracted.body.slice(0, 8000),
                urls: extracted.links,
                headers: '',
                language: extracted.language || 'VI',
            };
            const { data, usedMock: mock } = await analyzeMessage(payload, extracted.message_type || 'email');
            setResult(data);

            // Log detailed layer results to console for user management
            console.log(
                '%c[PhishGuard] Layer-by-Layer Detection Results:',
                'color: #ffffff; background: #262626; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;'
            );
            console.log(
                `%cVerdict: %c${data.label.toUpperCase()} %c| Risk Score: ${Math.round(data.risk_score * 100)}% | Confidence: ${Math.round(data.confidence * 100)}%`,
                'font-weight: bold;',
                `color: ${data.label === 'phishing' ? '#ef4444' : data.label === 'suspicious' ? '#f59e0b' : '#10b981'}; font-weight: bold;`,
                'color: #595959;'
            );
            if (data.layer_scores || data.layer_labels) {
                console.table(
                    Object.keys(data.layer_scores || {}).map((layer) => ({
                        'Layer Name': layer.toUpperCase(),
                        'Risk Score': data.layer_scores?.[layer] ?? 'N/A',
                        'Verdict Label': data.layer_labels?.[layer] ?? 'N/A',
                        'Fusion Weight': data.weights?.[layer] ? `${Math.round(data.weights[layer] * 100)}%` : 'N/A'
                    }))
                );
            } else {
                console.log('%cNo layer-by-layer breakdown available.', 'color: #8c8c8c; font-style: italic;');
            }

            setUsedMock(mock);
            setBackendOnline(!mock);
            setState('result');

            // Highlight evidence trên trang email
            triggerHighlight(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Đã xảy ra lỗi.');
            setState('error');
        }
    }, [tabId, triggerHighlight]);

    const handleReset = useCallback(() => {
        setState('idle');
        setError(null);
        setResult(null);

        // Xoá highlight trên trang email
        triggerClearHighlight();

        if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
            chrome.storage.local.remove([
                'phishguard_popup_state',
                'phishguard_popup_result',
                'phishguard_popup_error',
                'phishguard_popup_mock'
            ]);
        }
    }, [triggerClearHighlight]);

    const sc = result ? STATUS_CONFIG[result.label] ?? STATUS_CONFIG.unknown : null;

    return (
        <div style={{
            width: '100%',
            height: '100%',
            background: '#FFFFFF',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            display: 'flex', flexDirection: 'column', overflow: 'hidden',
            borderRadius: 16,
        }}>
            <style>{`
                @keyframes phishguard-fade-in {
                    from { opacity: 0; transform: translateY(6px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .animate-fade-in {
                    animation: phishguard-fade-in 0.3s ease-out forwards;
                }
                ::-webkit-scrollbar {
                    width: 5px;
                }
                ::-webkit-scrollbar-track {
                    background: transparent;
                }
                ::-webkit-scrollbar-thumb {
                    background: #E0E0E0;
                    border-radius: 4px;
                }
                ::-webkit-scrollbar-thumb:hover {
                    background: #CCCCCC;
                }
            `}</style>

            {/* ── Header ── */}
            <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                padding: '12px 16px 10px', 
                borderBottom: '1px solid #F0F0F0',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <ShieldIcon size={20} color="#434343" />
                    <div>
                        <div style={{ fontSize: 14, fontWeight: 700, color: '#1F1F1F', lineHeight: 1.1 }}>PhishGuard</div>
                        <div style={{ fontSize: 10, color: '#ACACAC', marginTop: 2 }}>Kiểm tra email lừa đảo</div>
                    </div>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {/* Nút Reset nhanh ở góc trên bên phải khi không ở trạng thái Idle */}
                    {state !== 'idle' && (
                        <button 
                            onClick={handleReset}
                            title="Quét tin nhắn mới"
                            style={{
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                width: 26, height: 26, borderRadius: 6, border: '1px solid #E8E8E8',
                                background: '#FFFFFF', color: '#8C8C8C', cursor: 'pointer',
                                transition: 'all 0.15s ease', outline: 'none',
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.borderColor = '#434343';
                                e.currentTarget.style.color = '#434343';
                                e.currentTarget.style.background = '#F5F5F5';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.borderColor = '#E8E8E8';
                                e.currentTarget.style.color = '#8C8C8C';
                                e.currentTarget.style.background = '#FFFFFF';
                            }}
                        >
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
                            </svg>
                        </button>
                    )}

                    {/* Nút Close (X) hiển thị khi ở dạng Floating Panel trong content script */}
                    {onClose && (
                        <button 
                            onClick={onClose}
                            title="Đóng"
                            style={{
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                width: 26, height: 26, borderRadius: 6, border: '1px solid #E8E8E8',
                                background: '#FFFFFF', color: '#8C8C8C', cursor: 'pointer',
                                transition: 'all 0.15s ease', outline: 'none',
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.borderColor = '#FF4D4F';
                                e.currentTarget.style.color = '#FF4D4F';
                                e.currentTarget.style.background = '#FFF1F0';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.borderColor = '#E8E8E8';
                                e.currentTarget.style.color = '#8C8C8C';
                                e.currentTarget.style.background = '#FFFFFF';
                            }}
                        >
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    )}
                </div>
            </div>

            {/* ── Body ── */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>

                {/* IDLE */}
                {state === 'idle' && (
                    <div className="animate-fade-in" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '28px 24px 24px', gap: 16 }}>
                        <div style={{
                            padding: 14, borderRadius: 16, background: '#F5F5F5',
                        }}>
                            <ShieldIcon size={44} color="#434343" />
                        </div>
                        <div style={{ textAlign: 'center' }}>
                            <h3 style={{ margin: '0 0 6px', fontSize: 15, fontWeight: 700, color: '#1F1F1F' }}>Sẵn sàng kiểm tra</h3>
                            <p style={{ fontSize: 12, color: '#8C8C8C', margin: 0, lineHeight: 1.5 }}>
                                Mở một email (Gmail, Outlook) hoặc tin nhắn SMS (Google Messages) để kiểm tra.
                            </p>
                        </div>
                        <button
                            onClick={handleDetect}
                            style={{
                                width: '100%', padding: '12px 0', borderRadius: 10, border: 'none',
                                background: '#262626',
                                color: '#FFFFFF', fontSize: 14, fontWeight: 600,
                                cursor: 'pointer', fontFamily: 'inherit',
                                transition: 'all 0.15s',
                            }}
                            onMouseEnter={(e) => { 
                                e.currentTarget.style.background = '#404040';
                            }}
                            onMouseLeave={(e) => { 
                                e.currentTarget.style.background = '#262626';
                            }}
                        >
                            Kiểm tra tin nhắn
                        </button>
                    </div>
                )}

                {/* LOADING */}
                {state === 'loading' && (
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px 24px', gap: 14 }}>
                        <SpinnerIcon />
                        <div style={{ textAlign: 'center' }}>
                            <p style={{ fontSize: 13, color: '#1F1F1F', margin: 0, fontWeight: 600 }}>Đang phân tích...</p>
                            <p style={{ fontSize: 11, color: '#ACACAC', margin: '4px 0 0' }}>Vui lòng chờ trong giây lát</p>
                        </div>
                    </div>
                )}

                {/* RESULT */}
                {state === 'result' && result && sc && (
                    <div className="animate-fade-in" style={{ flex: 1, display: 'flex', flexDirection: 'column', paddingBottom: 12 }}>
                        {/* Status card */}
                        <div style={{
                            margin: '12px 16px 0', padding: 14, borderRadius: 12,
                            background: sc.bg, border: `1px solid ${sc.border}`,
                            display: 'flex', alignItems: 'center', gap: 12,
                        }}>
                            <StatusIllustration type={sc.icon} />
                            <div style={{ flex: 1 }}>
                                <div style={{ fontSize: 20, fontWeight: 800, color: sc.textColor, lineHeight: 1.1 }}>{sc.label}</div>
                                <div style={{ fontSize: 12, color: '#595959', marginTop: 4, lineHeight: 1.4 }}>
                                    {result.xai_report?.summary || sc.sublabel}
                                </div>
                            </div>
                        </div>

                        {/* Evidence list */}
                        <EvidenceList result={result} />

                        {/* Recommendation */}
                        <RecommendationBox result={result} />

                        {/* Action buttons */}
                        <div style={{ padding: '10px 16px 0' }}>
                            <button
                                onClick={handleReset}
                                style={{
                                    width: '100%', padding: '9px 0', borderRadius: 8,
                                    border: '1px solid #D9D9D9', background: '#FFFFFF',
                                    color: '#595959', fontSize: 12, fontWeight: 500, cursor: 'pointer',
                                    fontFamily: 'inherit', transition: 'all 0.15s',
                                }}
                                onMouseEnter={(e) => { 
                                    e.currentTarget.style.background = '#FAFAFA'; 
                                    e.currentTarget.style.borderColor = '#BFBFBF';
                                }}
                                onMouseLeave={(e) => { 
                                    e.currentTarget.style.background = '#FFFFFF'; 
                                    e.currentTarget.style.borderColor = '#D9D9D9';
                                }}
                            >
                                Kiểm tra tin nhắn khác
                            </button>
                        </div>

                        {/* Technical details (collapsed) */}
                        <TechnicalDetails result={result} />
                    </div>
                )}

                {/* ERROR */}
                {state === 'error' && (
                    <div className="animate-fade-in" style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '24px 20px', gap: 12 }}>
                        <div style={{ padding: '12px 14px', borderRadius: 10, border: '1px solid #FFCCC7', background: '#FFF1F0', width: '100%' }}>
                            <p style={{ margin: 0, fontSize: 13, fontWeight: 700, color: '#CF1322' }}>Không thể phân tích</p>
                            <p style={{ margin: '4px 0 0', fontSize: 11, color: '#595959', lineHeight: 1.5 }}>
                                {error ?? 'Đã xảy ra lỗi.'}
                            </p>
                        </div>
                        <button
                            onClick={handleDetect}
                            style={{
                                width: '100%', padding: '11px 0', borderRadius: 10, border: 'none',
                                background: '#262626', color: '#FFFFFF', fontSize: 13, fontWeight: 600,
                                cursor: 'pointer', fontFamily: 'inherit',
                            }}
                        >
                            Thử lại
                        </button>
                        <button 
                            onClick={handleReset} 
                            style={{ 
                                background: 'none', border: 'none', cursor: 'pointer', 
                                fontSize: 11, color: '#8C8C8C', fontFamily: 'inherit',
                                transition: 'color 0.15s',
                            }}
                            onMouseEnter={(e) => { e.currentTarget.style.color = '#595959'; }}
                            onMouseLeave={(e) => { e.currentTarget.style.color = '#8C8C8C'; }}
                        >
                            Huỷ
                        </button>
                    </div>
                )}
            </div>

            {/* ── Footer ── */}
            <div style={{ 
                padding: '8px 16px', 
                borderTop: '1px solid #F0F0F0', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ 
                        width: 6, height: 6, borderRadius: '50%', 
                        background: usedMock ? '#FAAD14' : backendOnline ? '#52C41A' : '#FF4D4F', 
                        display: 'inline-block',
                    }} />
                    <span style={{ fontSize: 10, color: '#ACACAC', fontWeight: 500 }}>
                        {usedMock ? 'Demo' : backendOnline ? 'Đã kết nối' : 'Mất kết nối'}
                    </span>
                </div>
                <span style={{ fontSize: 9, color: '#D9D9D9', fontFamily: 'monospace' }}>v2.0</span>
            </div>
        </div>
    );
}
