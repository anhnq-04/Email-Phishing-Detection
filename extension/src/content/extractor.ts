import type { ExtractedMessage } from '@/src/types/analysis';

function extractGmailSender(): string {
    const senders = document.querySelectorAll('.gD');
    if (senders.length > 0) {
        const lastSender = senders[senders.length - 1];
        return lastSender.getAttribute('email') || lastSender.textContent?.trim() || '';
    }
    return '';
}

function extractGmailSubject(): string {
    const el = document.querySelector('h2.hP');
    if (el && el.textContent) {
        return el.textContent.trim();
    }
    return document.title.replace(' - Gmail', '').trim();
}

function extractGmailBody(): string {
    const bodies = document.querySelectorAll('.a3s.aiL');
    if (bodies.length > 0) {
        const lastBody = bodies[bodies.length - 1] as HTMLElement;
        return lastBody.innerText || lastBody.textContent || '';
    }
    return '';
}

function extractGmailLinks(): string[] {
    const bodies = document.querySelectorAll('.a3s.aiL');
    if (bodies.length > 0) {
        const lastBody = bodies[bodies.length - 1];
        const bodyText = (lastBody as HTMLElement).innerText || '';
        const anchors = Array.from(lastBody.querySelectorAll('a[href]'));
        return anchors
            .map((a) => (a as HTMLAnchorElement).href)
            .filter((href) => {
                if (!href.startsWith('http')) return false;

                // Skip mailto: links
                if (href.startsWith('mailto:')) return false;

                // Skip Gmail autolinked email domains:
                // Gmail auto-creates <a href="http://dut.udn.vn/"> for "@dut.udn.vn" in text.
                // These are bare domain URLs with no meaningful path.
                try {
                    const url = new URL(href);
                    const bareDomain = url.hostname;
                    // If the email body contains "@domain" but NOT "http(s)://domain",
                    // this is a Gmail autolink, not a real link in the email.
                    if (
                        url.pathname === '/' &&
                        !url.search &&
                        bodyText.includes('@' + bareDomain) &&
                        !bodyText.includes('://' + bareDomain)
                    ) {
                        return false;
                    }
                } catch {
                    // invalid URL, skip filtering
                }

                return true;
            })
            .slice(0, 50);
    }
    return [];
}

function extractOutlookSender(): string {
    const el =
        document.querySelector('[data-lpc-hover-target-id]') ||
        document.querySelector('.ms-Persona-primaryText') ||
        document.querySelector('[aria-label*="From"]');
    return el?.textContent?.trim() ?? '';
}

function extractOutlookBody(): string {
    const body = document.querySelector('[aria-label="Message body"]') || document.querySelector('[role="document"]');
    return (body as HTMLElement)?.innerText || (body as HTMLElement)?.textContent || '';
}

function extractOutlookLinks(): string[] {
    const body = document.querySelector('[aria-label="Message body"]') || document.querySelector('[role="document"]');
    if (body) {
        const anchors = Array.from(body.querySelectorAll('a[href]'));
        return anchors
            .map((a) => (a as HTMLAnchorElement).href)
            .filter((href) => href.startsWith('http') || href.startsWith('https'))
            .slice(0, 50);
    }
    return [];
}

function extractFacebookSender(): string {
    const el =
        document.querySelector('h2[data-text]') ||
        document.querySelector('._2eke') ||
        document.querySelector('[data-testid="conversation_name"]');
    return el?.textContent?.trim() ?? '';
}

function extractSmsSender(): string {
    const host = window.location.hostname;
    
    if (host.includes('messages.google.com')) {
        const headerSelectors = [
            'mws-header [data-e2e-header-title] h2',
            '[data-e2e-header-title] h2',
            '[data-e2e-conversation-header] h2',
            'mws-header h2',
            'mws-conversation-header .name',
            'mws-conversation-header h1',
            'mws-conversation-header h2',
            '.conversation-title',
            '.contact-name',
            'h1.name',
            'h2.name'
        ];
        
        for (const selector of headerSelectors) {
            const el = document.querySelector(selector);
            const text = el?.textContent?.trim();
            if (text) return text;
        }

        return document.title.replace(' - Messages', '').replace('Messages', '').trim();
    }
    
    const el =
        document.querySelector('.conversation-from') ||
        document.querySelector('.message-sender') ||
        document.querySelector('[data-sender]');
    return el?.textContent?.trim() ?? '';
}

function extractSmsBody(): string {
    const host = window.location.hostname;
    
    if (host.includes('messages.google.com')) {
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
                const lastText = bubbles[bubbles.length - 1].textContent?.trim();
                if (lastText) return lastText;
            }
        }

        // Fallback for newer Angular class structures
        const starInsertedDivs = Array.from(document.querySelectorAll('div.ng-star-inserted'));
        for (let i = starInsertedDivs.length - 1; i >= 0; i--) {
            const txt = starInsertedDivs[i].textContent?.trim() || '';
            if (txt.length > 15 && txt.length < 500 && !txt.includes('Messages') && !txt.includes('Settings')) {
                return txt;
            }
        }
    }

    const genericBody = document.querySelector('.message-body') || document.querySelector('.sms-body') || document.querySelector('.message-content');
    return genericBody?.textContent?.trim() ?? '';
}

function extractGoogleMessagesLinks(): string[] {
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
            const lastBubble = bubbles[bubbles.length - 1];
            const anchors = Array.from(lastBubble.querySelectorAll('a[href]'));
            return anchors
                .map((a) => (a as HTMLAnchorElement).href)
                .filter((href) => href.startsWith('http') || href.startsWith('https'))
                .slice(0, 50);
        }
    }
    
    return [];
}

function detectLanguage(text: string): 'VI' | 'EN' {
    const viAccents = /[áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]/i;
    
    let accentCount = 0;
    for (const char of text) {
        if (viAccents.test(char)) accentCount++;
    }

    const viStopWords = /\b(và|của|được|là|trong|cho|với|có|không|tôi|bạn|này|đó|để|nội|dung|xác|nhận)\b/gi;
    const enStopWords = /\b(the|of|and|to|in|is|that|for|it|with|as|was|on|at|by|this|your|you)\b/gi;

    const viWordCount = (text.match(viStopWords) || []).length;
    const enWordCount = (text.match(enStopWords) || []).length;

    if (accentCount > 0 || viWordCount > enWordCount) {
        return 'VI';
    }
    return 'EN';
}

function detectSender(): string {
    const host = window.location.hostname;
    if (host.includes('mail.google.com')) return extractGmailSender();
    if (host.includes('outlook') || host.includes('office.com')) return extractOutlookSender();
    if (host.includes('facebook.com') || host.includes('messenger.com'))
        return extractFacebookSender();
    return extractSmsSender();
}

function extractVisibleText(): string {
    const skipTags = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'IFRAME', 'HEAD']);
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
        acceptNode: (node) => {
            const parent = node.parentElement;
            if (!parent) return NodeFilter.FILTER_REJECT;
            if (skipTags.has(parent.tagName)) return NodeFilter.FILTER_REJECT;
            const style = window.getComputedStyle(parent);
            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0')
                return NodeFilter.FILTER_REJECT;
            const text = node.textContent?.trim();
            if (!text) return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
        },
    });

    const parts: string[] = [];
    let node: Node | null;
    while ((node = walker.nextNode())) {
        const text = node.textContent?.trim();
        if (text) parts.push(text);
    }
    return parts.join(' ').slice(0, 8000);
}

function extractLinks(): string[] {
    const anchors = Array.from(document.querySelectorAll('a[href]'));
    return anchors
        .map((a) => (a as HTMLAnchorElement).href)
        .filter((href) => href.startsWith('http') || href.startsWith('https'))
        .slice(0, 50);
}

function extractActiveGmailMessage(): ExtractedMessage | null {
    const messages = Array.from(document.querySelectorAll('.adn'));
    // Filter for expanded messages. An expanded message has a visible body (.a3s).
    const expandedMessages = messages.filter((msg) => {
        const body = msg.querySelector('.a3s.aiL');
        return body && ((body as HTMLElement).clientHeight > 0 || (body as HTMLElement).offsetHeight > 0);
    });

    // If there is an expanded message, use the last one (most recent expanded)
    const activeMessage = expandedMessages.length > 0 ? expandedMessages[expandedMessages.length - 1] : null;

    if (activeMessage) {
        const senderEl = activeMessage.querySelector('.gD');
        const sender = senderEl?.getAttribute('email') || senderEl?.textContent?.trim() || '';

        const subjectEl = document.querySelector('h2.hP');
        const subject = subjectEl?.textContent?.trim() || document.title.replace(' - Gmail', '').trim();

        const bodyEl = activeMessage.querySelector('.a3s.aiL') as HTMLElement;
        const body = bodyEl?.innerText || bodyEl?.textContent || '';

        // Extract links within this specific message body
        const anchors = Array.from(bodyEl.querySelectorAll('a[href]'));
        const bodyText = bodyEl.innerText || '';
        const links = anchors
            .map((a) => (a as HTMLAnchorElement).href)
            .filter((href) => {
                if (!href.startsWith('http')) return false;
                if (href.startsWith('mailto:')) return false;
                try {
                    const url = new URL(href);
                    const bareDomain = url.hostname;
                    if (
                        url.pathname === '/' &&
                        !url.search &&
                        bodyText.includes('@' + bareDomain) &&
                        !bodyText.includes('://' + bareDomain)
                    ) {
                        return false;
                    }
                } catch {}
                return true;
            })
            .slice(0, 50);

        return { sender, subject, body, links };
    }

    return null;
}

export function extractMessage(): ExtractedMessage {
    const host = window.location.hostname;
    let result: ExtractedMessage;

    if (host.includes('mail.google.com') || host.includes('gmail.com')) {
        const activeMsg = extractActiveGmailMessage();
        if (activeMsg && (activeMsg.body || activeMsg.sender)) {
            result = activeMsg;
        } else {
            const sender = extractGmailSender();
            const subject = extractGmailSubject();
            const body = extractGmailBody();
            const links = extractGmailLinks();
            result = { sender, subject, body, links };
        }
        result.message_type = 'email';
    } else if (host.includes('outlook') || host.includes('office.com') || host.includes('office365')) {
        const sender = extractOutlookSender();
        const subject = document.title.replace(' - Outlook', '').trim();
        const body = extractOutlookBody();
        const links = extractOutlookLinks();
        result = { sender, subject, body, links };
        result.message_type = 'email';
    } else if (host.includes('messages.google.com')) {
        // Kiểm tra xem có đang ở màn hình landing (chưa mở cuộc trò chuyện nào) hay không
        const isLanding = document.querySelector('.landing-img') || document.querySelector('mws-icon.landing-img') || document.querySelector('.landing-container');
        if (isLanding) {
            result = { sender: '', subject: 'SMS Message', body: '', links: [] };
        } else {
            const sender = extractSmsSender();
            const subject = 'SMS Message';
            const body = extractSmsBody();
            const links = extractGoogleMessagesLinks();
            result = { sender, subject, body, links };
        }
        result.message_type = 'sms';
    } else {
        const sender = detectSender();
        const subject = document.title || '';
        const body = extractVisibleText();
        const links = extractLinks();
        result = { sender, subject, body, links };
        result.message_type = host.includes('facebook.com') || host.includes('messenger.com') ? 'sms' : 'email';
    }

    // Detect language of the message body dynamically
    result.language = detectLanguage(result.body);

    return result;
}
