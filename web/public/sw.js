// Replaced with an asset fingerprint by the build; updates wait for old app windows to close.
const CACHE = "weedwise-offline-__WEEDWISE_CACHE_VERSION__";
const OFFLINE = new URL("./offline.html", self.registration.scope).href;
const ICON = new URL("./weedwise-logo.svg", self.registration.scope).href;

self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll([OFFLINE, ICON])));
});

self.addEventListener("activate", event => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter(name => name.startsWith("weedwise-offline-") && name !== CACHE)
      .map(name => caches.delete(name)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", event => {
  const {request} = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;
  // Photos, predictions, chat replies and API credentials never enter this cache.
  if (request.mode === "navigate" && (url.pathname === "/" || url.pathname === "/index.html")) {
    event.respondWith(fetch(request).catch(async () => {
      const cache = await caches.open(CACHE);
      return await cache.match(OFFLINE) || Response.error();
    }));
  } else if (url.href === ICON) {
    event.respondWith(fetch(request).catch(async () => {
      const cache = await caches.open(CACHE);
      return await cache.match(ICON) || Response.error();
    }));
  }
});
