const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const os = require("node:os");
const {pathToFileURL} = require("node:url");
const {chromium} = require("playwright");

(async () => {
  const dist = path.resolve(__dirname, "../../web/dist");
  let server;
  let base = process.argv[2];
  if (!base) {
    process.chdir(dist);
    const types = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
      ".webmanifest": "application/manifest+json", ".svg": "image/svg+xml", ".jpg": "image/jpeg", ".png": "image/png"};
    server = http.createServer(async (req, res) => {
      if (req.url === "/api/chat") {
        res.setHeader("Content-Type", "application/json");
        res.end('{"configured":false}');
        return;
      }
      if (req.url === "/api/predict") {
        try {
          const predict = (await import(pathToFileURL(path.join(dist, "api/predict.js")))).default;
          const parts = [];
          for await (const part of req) parts.push(part);
          req.body = JSON.parse(Buffer.concat(parts).toString());
          res.status = code => {res.statusCode = code; return res;};
          res.json = value => {res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(value));};
          await predict(req, res);
        } catch (error) {
          console.error(error);
          res.writeHead(500).end();
        }
        return;
      }
      const pathname = new URL(req.url, "http://localhost").pathname;
      const file = path.join(dist, "public", pathname === "/" ? "index.html" : pathname);
      if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {res.writeHead(404).end(); return;}
      res.setHeader("Content-Type", types[path.extname(file)] || "application/octet-stream");
      res.setHeader("Cache-Control", "no-cache");
      fs.createReadStream(file).pipe(res);
    });
    base = await new Promise(resolve => server.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${server.address().port}/`)));
  }
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "weedwise-pwa-test-"));
  const context = await chromium.launchPersistentContext(profile, {channel: "msedge", headless: true,
    args: ["--disable-http2", "--disable-quic"], viewport: {width: 390, height: 844}});
  try {
    const page = await context.newPage();
    await page.goto(base, {waitUntil: "load"});
    await page.evaluate(() => navigator.serviceWorker.ready);
    await page.waitForFunction(() => navigator.serviceWorker.controller !== null);
    const manifestLink = await page.locator('link[rel="manifest"]').getAttribute("href");
    const response = await context.request.get(new URL(manifestLink, base).href);
    assert(response.ok());
    assert.match(response.headers()["content-type"], /application\/(manifest\+)?json/);
    const manifest = await response.json();
    assert.equal(manifest.display, "standalone");
    assert.equal(manifest.start_url, "/");
    assert.equal(manifest.scope, "/");
    assert.equal(manifest.short_name, "WeedWise");
    for (const icon of [...manifest.icons, {src: "/icons/apple-touch-icon.png", sizes: "180x180"}]) {
      const dimensions = await page.evaluate(src => new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(`${img.naturalWidth}x${img.naturalHeight}`);
        img.onerror = reject;
        img.src = src;
      }), icon.src);
      assert.equal(dimensions, icon.sizes);
    }
    const cdp = await context.newCDPSession(page);
    const appManifest = await cdp.send("Page.getAppManifest");
    assert.equal(appManifest.errors.length, 0, JSON.stringify(appManifest.errors));
    const eligibility = await cdp.send("Page.getInstallabilityErrors");
    assert.deepEqual(eligibility.installabilityErrors, []);
    console.log("PASS manifest, PNG icons, service-worker control and browser installability criteria");

    // Force the manual-install path even if the browser has already offered its native prompt.
    const fallbackPage = await context.newPage();
    await fallbackPage.addInitScript(() => window.addEventListener("beforeinstallprompt", event => {
      event.preventDefault(); event.stopImmediatePropagation();
    }));
    await fallbackPage.goto(base, {waitUntil: "load"});
    await fallbackPage.getByRole("button", {name: "Install WeedWise", exact: true}).click();
    assert(await fallbackPage.locator("#pwa-install-help").evaluate(el => el.open));
    await fallbackPage.close();
    // Exercise both native-prompt outcomes without installing an app on the test machine.
    for (const outcome of ["dismissed", "accepted"]) {
      await page.evaluate(outcome => {
        const event = new Event("beforeinstallprompt", {cancelable: true});
        event.prompt = async () => { window.promptCalls = (window.promptCalls || 0) + 1; };
        event.userChoice = Promise.resolve({outcome});
        window.dispatchEvent(event);
      }, outcome);
      await page.getByRole("button", {name: "Install WeedWise", exact: true}).click();
      await page.waitForFunction(() => !document.getElementById("pwa-install-button").disabled);
      assert.match(await page.locator("#pwa-install-status").innerText(), outcome === "accepted" ? /Installation accepted/ : /later/);
    }
    assert.equal(await page.evaluate(() => window.promptCalls), 2);
    await page.evaluate(() => window.dispatchEvent(new Event("appinstalled")));
    assert(await page.locator("#install").isHidden());
    console.log("PASS manual install help and install-prompt UI states");

    await page.reload({waitUntil: "load"});
    for (const width of [320, 390, 768, 1440]) {
      await page.setViewportSize({width, height: 900});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    }
    await page.setViewportSize({width: 390, height: 844});
    await page.locator("#install").screenshot({path: path.join(os.tmpdir(), "weedwise-pwa-mobile.png")});
    await page.getByRole("tab", {name: "Examples", exact: true}).click();
    await page.locator('[data-example="maize.jpg"]').click();
    await page.getByText("Photo ready.", {exact: true}).waitFor();
    await page.getByRole("button", {name: "Run identification"}).click();
    await page.getByText("Prediction complete.", {exact: true}).waitFor({timeout: 60000});
    assert.equal(await page.locator("#prediction").innerText(), "Maize");
    assert.equal(await page.locator("#ranking li").count(), 5);
    await page.getByRole("button", {name: "Open chat assistant"}).click();
    await page.getByRole("dialog").waitFor();
    await page.getByRole("button", {name: "Close chat", exact: true}).click();
    console.log("PASS responsive layout, real model prediction and floating chat under service-worker control");

    await context.setOffline(true);
    await page.locator("#pwa-offline").waitFor({state: "visible"});
    await page.reload({waitUntil: "load"});
    assert.equal(await page.title(), "WeedWise | You’re offline");
    assert(await page.locator("img").evaluate(img => img.complete && img.naturalWidth > 0));
    await page.screenshot({path: path.join(os.tmpdir(), "weedwise-pwa-offline.png")});
    const cacheEntries = await page.evaluate(async () => {
      const entries = [];
      for (const name of await caches.keys()) {
        if (name.startsWith("weedwise-offline-")) {
          const cache = await caches.open(name);
          entries.push(...(await cache.keys()).map(req => new URL(req.url).pathname));
        }
      }
      return entries.sort();
    });
    assert.deepEqual(cacheEntries, ["/offline.html", "/weedwise-logo.svg"]);
    await context.setOffline(false);
    await page.getByRole("link", {name: "Try again", exact: true}).click();
    await page.locator(".hero h1").waitFor();
    await page.locator("#pwa-offline").waitFor({state: "hidden"});
    const missing = await context.request.get(new URL("/nonexistent-pwa-test-route", base).href);
    assert.equal(missing.status(), 404);
    console.log("PASS offline fallback, cached logo, online recovery, API-cache exclusion and unchanged 404 routes");
  } finally {
    await context.close();
    const tempRoot = path.resolve(os.tmpdir()) + path.sep;
    assert(path.resolve(profile).startsWith(tempRoot) && path.basename(profile).startsWith("weedwise-pwa-test-"));
    fs.rmSync(profile, {recursive: true, force: true, maxRetries: 5});
    if (server) {server.closeAllConnections(); await new Promise(resolve => server.close(resolve));}
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
