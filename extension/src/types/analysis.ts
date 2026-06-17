export type VerdictLabel = 'legit' | 'suspicious' | 'phishing' | 'unknown';
export type ActionType = 'allow' | 'warn' | 'block' | 'review';
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';
export type Severity = 'critical' | 'high' | 'medium' | 'low';

export interface EvidenceItem {
    category: string;
    signal: string;
    severity: Severity;
    title: string;
    description: string;
    matched_text: string;
    source_layer: string;
}

export interface XaiReport {
    summary: string;
    evidence_items: EvidenceItem[];
    recommendation?: string;
    layer_explanations?: Record<string, string>;
    top_signals?: string[];
}

export interface AnalysisResponse {
    message_type: string;
    label: VerdictLabel;
    risk_score: number;
    risk_level: RiskLevel;
    action: ActionType;
    confidence: number;
    reason: string;
    signals: string[];
    evidence?: string[];
    weights?: Record<string, number>;
    layer_scores?: Record<string, number>;
    layer_labels?: Record<string, string>;
    disagreement?: number;
    xai_report?: XaiReport;
}

export interface AnalysisRequest {
    sender_email: string;
    subject: string;
    body: string;
    urls?: string[];
    headers: string;
    language: string;
}

export interface ExtractedMessage {
    sender: string;
    subject: string;
    body: string;
    links: string[];
    language?: 'VI' | 'EN';
    message_type?: 'email' | 'sms';
}

export type PopupState = 'idle' | 'loading' | 'result' | 'error';

export interface PopupData {
    state: PopupState;
    result: AnalysisResponse | null;
    error: string | null;
    usedMock: boolean;
}
