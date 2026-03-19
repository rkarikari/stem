// coi-serviceworker.js — Cross-Origin Isolation Service Worker
// Place this file in the same directory as chess.html and commit both to GitHub Pages.
// Based on: github.com/gzuidhof/coi-serviceworker (MIT License)
// This SW intercepts all responses and injects COOP/COEP headers,
// enabling SharedArrayBuffer → full multi-threaded Stockfish 18.

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', e => e.waitUntil(
  self.clients.claim().then(() =>
    self.clients.matchAll().then(clients =>
      clients.forEach(c => c.postMessage({type: 'COI_ACTIVATED'}))
    )
  )
));

self.addEventListener('message', e => {
  if (e.data && e.data.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('fetch', e => {
  const req = e.request;
  // Don't intercept non-GET or opaque requests
  if (req.method !== 'GET') return;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;

  e.respondWith(
    fetch(req).then(res => {
      // Don't modify error/opaque responses
      if (!res || res.status === 0 || !res.ok && res.status !== 0) {
        // Still add headers if possible
        if (!res || res.type === 'opaque') return res;
      }
      const headers = new Headers(res.headers);
      headers.set('Cross-Origin-Opener-Policy',   'same-origin');
      headers.set('Cross-Origin-Embedder-Policy',  'credentialless');
      headers.set('Cross-Origin-Resource-Policy',  'cross-origin');
      return new Response(res.body, {
        status:     res.status,
        statusText: res.statusText,
        headers:    headers,
      });
    }).catch(() => fetch(req))
  );
});
