import type { AnalysisRequest, AnalysisResponse } from '@/src/types/analysis';

const BASE_URL = 'https://ngoquocanh1501--graduation-project-app-phishingapi-web.modal.run';
const API_URL = `${BASE_URL}/api/v1/emails/analyze`;
const HEALTH_URL = `${BASE_URL}/api/v1/health`;

export const MOCK_RESPONSE: AnalysisResponse = {
    message_type: 'email',
    label: 'phishing',
    risk_score: 0.91,
    risk_level: 'high',
    action: 'block',
    confidence: 0.85,
    reason: 'Email contains credential request and suspicious domain.',
    signals: ['credential_request', 'domain_spoofing'],
    xai_report: {
        summary:
            'Email is suspicious because it asks for account verification and contains a risky link.',
        evidence_items: [
            {
                category: 'content',
                signal: 'credential_request',
                severity: 'critical',
                title: 'Credential request',
                description: 'The message asks the user to verify or provide account information.',
                matched_text: 'verify your account',
                source_layer: 'xai',
            },
            {
                category: 'url',
                signal: 'domain_spoofing',
                severity: 'critical',
                title: 'Suspicious domain',
                description: 'The URL domain does not match the claimed organization.',
                matched_text: 'https://fake-login.example.com',
                source_layer: 'url',
            },
        ],
    },
};

export async function checkBackendStatus(): Promise<boolean> {
    try {
        const res = await fetch(HEALTH_URL, {
            method: 'GET',
            signal: AbortSignal.timeout(30_000),
        });
        return res.ok;
    } catch {
        return false;
    }
}

export async function analyzeMessage(
    payload: AnalysisRequest,
    messageType: 'email' | 'sms' = 'email'
): Promise<{ data: AnalysisResponse; usedMock: boolean }> {
    try {
        const endpoint = messageType === 'sms'
            ? `${BASE_URL}/api/v1/sms/analyze`
            : `${BASE_URL}/api/v1/emails/analyze`;

        const requestBody = messageType === 'sms'
            ? {
                sender: payload.sender_email,
                content: payload.body,
                urls: payload.urls,
                language: payload.language
            }
            : payload;

        console.log(`[PhishGuard] Sending ${messageType} payload:`, JSON.stringify(requestBody).slice(0, 500));

        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
            signal: AbortSignal.timeout(120_000),
        });

        if (!res.ok) {
            console.error(`[PhishGuard] API error: HTTP ${res.status} ${res.statusText}`);
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        const data: AnalysisResponse = await res.json();
        console.log('[PhishGuard] API response:', JSON.stringify(data).slice(0, 800));
        return { data, usedMock: false };
    } catch (err) {
        console.error('[PhishGuard] API call failed, using mock:', err);
        return { data: MOCK_RESPONSE, usedMock: true };
    }
}
