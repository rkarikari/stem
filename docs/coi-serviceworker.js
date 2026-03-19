// coi-serviceworker.js — Cross-Origin Isolation Service Worker
// Place in same folder as chess.html. Commit both to GitHub Pages repo.
// Injects COOP/COEP headers so SharedArrayBuffer works → full Stockfish 18.

self.addEventListener('install', () => {
  console.log('[COI-SW] installed');
  self.skipWaiting(); // activate immediately, don't wait for old tabs to close
});

self.addEventListener('activate', e => {
  console.log('[COI-SW] activated');
  // claim() makes this SW control all open clients immediately
  // without requiring a page reload from the client side
  e.waitUntil(self.clients.claim());
});

self.addEventListener('message', e => {
  if (e.data && e.data.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  // Don't break opaque requests (no-cors mode resources)
  if (req.mode === 'no-cors') return;

  e.respondWith(
    fetch(req)
      .then(res => {
        // Don't modify opaque or error responses
        if (!res || res.type === 'opaque' || res.type === 'error') return res;

        const headers = new Headers(res.headers);
        // These two headers together = crossOriginIsolated: true in the browser
        headers.set('Cross-Origin-Opener-Policy',  'same-origin');
        // 'credentialless' is more permissive than 'require-corp':
        // cross-origin resources (fonts, images) still load without CORP headers
        headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
        headers.set('Cross-Origin-Resource-Policy', 'cross-origin');

        return new Response(res.body, {
          status:     res.status,
          statusText: res.statusText,
          headers:    headers,
        });
      })
      .catch(() => fetch(req)) // network failure — just pass through
  );
});
