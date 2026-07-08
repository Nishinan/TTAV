// I: lightweight interaction telemetry.
//
// A local-only, append-only event log for user-study / case-study analysis.
// Nothing is uploaded — events live in a bounded in-memory ring buffer, are
// mirrored to localStorage so a page reload doesn't lose them, and can be
// exported as JSON. Instrumentation is wired centrally in plotView via the
// store's subscribeWithSelector middleware, so call sites stay untouched.

export type InteractionEvent = {
    t: number;            // wall-clock ms (Date.now)
    session: string;      // page-session id, groups events across reloads
    type: string;         // event kind, e.g. 'epoch_change'
    [key: string]: unknown;
};

const STORAGE_KEY = 'ttav_interaction_log';
const MAX_EVENTS = 5000;      // ring buffer cap; oldest dropped past this
const PERSIST_DEBOUNCE_MS = 500;

// A fresh id per page load so distinct sessions are separable in the export.
const SESSION_ID = `${new Date().toISOString().slice(0, 19)}-${Math.random().toString(36).slice(2, 8)}`;

function loadFromStorage(): InteractionEvent[] {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.slice(-MAX_EVENTS) : [];
    } catch {
        return [];
    }
}

let buffer: InteractionEvent[] = loadFromStorage();
let persistTimer: ReturnType<typeof setTimeout> | null = null;

function schedulePersist(): void {
    if (persistTimer !== null) return;
    persistTimer = setTimeout(() => {
        persistTimer = null;
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(buffer));
        } catch {
            // localStorage full or unavailable — drop silently, telemetry is best-effort.
        }
    }, PERSIST_DEBOUNCE_MS);
}

export function logInteraction(type: string, payload?: Record<string, unknown>): void {
    buffer.push({ t: Date.now(), session: SESSION_ID, type, ...(payload ?? {}) });
    if (buffer.length > MAX_EVENTS) {
        buffer = buffer.slice(-MAX_EVENTS);
    }
    schedulePersist();
}

export function getInteractionLog(): InteractionEvent[] {
    return buffer;
}

export function interactionLogSize(): number {
    return buffer.length;
}

export function clearInteractionLog(): void {
    buffer = [];
    try {
        localStorage.removeItem(STORAGE_KEY);
    } catch {
        // ignore
    }
}

export function exportInteractionLog(): void {
    const blob = new Blob([JSON.stringify(buffer, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `interaction-log-${new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// Best-effort flush so the last events survive a tab close mid-debounce.
if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(buffer));
        } catch {
            // ignore
        }
    });
}
