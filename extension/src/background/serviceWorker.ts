// ── Auto-inject content script into existing tabs on install/update ──
// When the extension is installed or reloaded, existing tabs don't have
// the content script injected yet. We manually inject it so the first
// "Check Message" click works without requiring a page reload.

chrome.runtime.onInstalled.addListener(async () => {
    console.log('[PhishGuard] Service worker installed/updated.');

    // Read content script path from manifest
    const mf = chrome.runtime.getManifest();
    const csEntry = mf.content_scripts?.[0];
    const csFiles = csEntry?.js;
    if (!csFiles || csFiles.length === 0) {
        console.warn('[PhishGuard] No content scripts found in manifest.');
        return;
    }

    // Inject into all existing tabs that match
    try {
        const tabs = await chrome.tabs.query({});
        for (const tab of tabs) {
            if (!tab.id || !tab.url) continue;
            // Skip internal chrome pages
            if (
                tab.url.startsWith('chrome://') ||
                tab.url.startsWith('chrome-extension://') ||
                tab.url.startsWith('edge://') ||
                tab.url.startsWith('about:')
            ) continue;

            try {
                await chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    files: csFiles,
                });
                console.log(`[PhishGuard] Injected into tab ${tab.id}`);
            } catch {
                // Some tabs can't be injected (web store, etc.)
            }
        }
    } catch (err) {
        console.warn('[PhishGuard] Tab injection error:', err);
    }
});

chrome.action.onClicked.addListener((tab) => {
    if (tab.id) {
        chrome.tabs.sendMessage(tab.id, { type: 'TOGGLE_FLOATING_PANEL' }).catch(() => {
            // Ignore error if tab content script is not yet loaded
        });
    }
});

chrome.runtime.onMessage.addListener((_message, _sender, _sendResponse) => {
    return false;
});

export {};
