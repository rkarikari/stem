// coi-serviceworker.js v1.012
// Injects COOP/COEP headers so SharedArrayBuffer works on GitHub Pages.

self.addEventListener('install', () => {
  console.log('[COI-SW] installed');
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  console.log('[COI-SW] activated');
  e.waitUntil(self.clients.claim());
});

self.addEventListener('message', e => {
  if (e.data && e.data.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('fetch', e => {
  const req  = e.request;
  const url  = req.url;

  // Only intercept GET requests
  if (req.method !== 'GET') return;

  // Skip opaque/no-cors requests — adding headers to them causes errors
  if (req.mode === 'no-cors') return;

  // Skip requests that aren't http/https (e.g. chrome-extension://, blob:, data:)
  if (!url.startsWith('http')) return;

  // Skip requests from blob: workers trying to fetch external resources —
  // these have a null/opaque origin and re-fetching them causes URL corruption.
  const origin = (e.request.referrer || '');
  if (origin.startsWith('blob:')) return;

  e.respondWith(
    fetch(req)
      .then(res => {
        if (!res || res.type === 'opaque' || res.type === 'error') return res;
        const headers = new Headers(res.headers);
        headers.set('Cross-Origin-Opener-Policy',  'same-origin');
        headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
        headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
        return new Response(res.body, {
          status:     res.status,
          statusText: res.statusText,
          headers:    headers,
        });
      })
      .catch(() => fetch(req))
  );
});
