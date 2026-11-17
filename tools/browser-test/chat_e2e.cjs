const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const os = require("node:os");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "../..");
const publicDir = path.join(root, "web/dist/public");
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".jpg": "image/jpeg", ".json": "application/json" };

async function main() {
  process.chdir(path.join(root, "web/dist"));
  const server = http.createServer(async (req, res) => {
    try {
      const pathname = new URL(req.url, "http://localhost").pathname;
      if (["/api/chat", "/api/predict"].includes(pathname)) {
        const handler = (await import(pathToFileURL(path.join(root, "web/dist", pathname.slice(1) + ".js")).href)).default;
        const chunks = [];
        for await (const chunk of req) chunks.push(chunk);
        req.body = chunks.length ? JSON.parse(Buffer.concat(chunks)) : undefined;
        res.status = code => { res.statusCode = code; return res; };
        res.json = data => { res.setHeader("content-type", "application/json"); res.end(JSON.stringify(data)); };
        await handler(req, res);
        return;
      }
      const target = path.resolve(publicDir, "." + (pathname === "/" ? "/index.html" : pathname));
      if (!target.startsWith(publicDir + path.sep) || !fs.existsSync(target)) { res.writeHead(404).end(); return; }
      res.setHeader("Content-Type", types[path.extname(target)] || "application/octet-stream");
      fs.createReadStream(target).pipe(res);
    } catch { res.writeHead(500).end(); }
  });
  const local = await new Promise(resolve => server.listen(0, "127.0.0.1", () => resolve("http://127.0.0.1:" + server.address().port)));
  const base = process.argv[2] || local;
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(base + "/#chat");
    await page.getByText("Chat is ready for setup.", { exact: false }).waitFor();
    await page.locator("#chat-input").fill("A question before setup");
    assert(await page.getByRole("button", { name: "Send message" }).isDisabled());
    await page.getByRole("button", { name: "Open chat assistant" }).click();
    await page.getByRole("dialog").waitFor();
    assert(await page.getByRole("dialog").getByText("A question before setup", { exact: true }).count() ||
      (await page.locator("#chat-input").inputValue()) === "A question before setup");
    await page.keyboard.press("Escape");
    await page.getByRole("dialog").waitFor({ state: "hidden" });
    console.log("PASS missing-key setup state and floating chat keyboard dismissal");

    let fail = false;
    let releaseDelayed;
    let delayStarted;
    const delayedStart = new Promise(resolve => { delayStarted = resolve; });
    const requests = [];
    await page.route("**/api/chat", async route => {
      if (route.request().method() === "GET") return route.fulfill({ json: { configured: true } });
      const body = route.request().postDataJSON();
      requests.push(body);
      if (fail) {
        fail = false;
        return route.fulfill({ status: 429, json: { error: "Test quota reached. Retry later." } });
      }
      if (body.messages.at(-1).content === "Delayed first question") {
        delayStarted();
        await new Promise(resolve => { releaseDelayed = resolve; });
      }
      return route.fulfill({ json: { answer: "Test response to " + body.messages.at(-1).content } });
    });
    await page.reload();
    await page.getByText("Ready for your questions.", { exact: true }).waitFor();
    await page.locator("#chat-input").fill("What is photosynthesis?");
    await page.getByRole("button", { name: "Send message" }).click();
    await page.getByText("Test response to What is photosynthesis?", { exact: true }).waitFor();
    await page.locator("#chat-input").fill("Explain it simply");
    await page.locator("#chat-input").press("Enter");
    await page.getByText("Test response to Explain it simply", { exact: true }).waitFor();
    assert.equal(requests.at(-1).messages.length, 3);
    await page.getByRole("button", { name: "New chat", exact: false }).click();
    await page.locator("#chat-input").fill("A separate conversation");
    await page.getByRole("button", { name: "Send message" }).click();
    await page.getByText("Test response to A separate conversation", { exact: true }).waitFor();
    assert.equal(requests.at(-1).messages.length, 1);
    assert.equal(await page.locator("#chat-history button").count(), 2);
    await page.locator("#chat-history").getByRole("button", { name: "What is photosynthesis?" }).click();
    assert.equal(await page.locator("#chat-messages article").count(), 4);
    await page.reload();
    await page.getByText("Ready for your questions.", { exact: true }).waitFor();
    assert.equal(await page.locator("#chat-history button").count(), 2);
    await page.locator("#chat-history").getByRole("button", { name: "What is photosynthesis?" }).click();
    assert.equal(await page.locator("#chat-messages article").count(), 4);
    console.log("PASS conversation context, separate threads and persistence after reload");

    fail = true;
    await page.locator("#chat-input").fill("Retry this question");
    await page.getByRole("button", { name: "Send message" }).click();
    await page.getByRole("alert").getByText("Test quota reached. Retry later.").waitFor();
    assert.equal(await page.locator("#chat-messages article").count(), 5);
    await page.getByRole("button", { name: "Retry last message" }).click();
    await page.getByText("Test response to Retry this question", { exact: true }).waitFor();
    assert.equal(await page.locator("#chat-messages article").count(), 6);
    console.log("PASS retry preserves user messages without duplicates");

    await page.getByRole("button", { name: "New chat", exact: false }).click();
    await page.locator("#chat-input").fill("Delayed first question");
    await page.getByRole("button", { name: "Send message" }).click();
    await delayedStart;
    await page.getByRole("button", { name: "New chat", exact: false }).click();
    await page.locator("#chat-input").fill("<img src=x onerror=alert(1)>");
    await page.getByRole("button", { name: "Send message" }).click();
    await page.getByText("Test response to <img src=x onerror=alert(1)>", { exact: true }).waitFor();
    releaseDelayed();
    await page.locator("#chat-history").getByRole("button", { name: "Delayed first question" }).click();
    await page.getByText("Test response to Delayed first question", { exact: true }).waitFor();
    assert.equal(await page.locator("#chat-messages img").count(), 0);
    console.log("PASS background replies stay in their own chat and message text is escaped");

    await page.getByRole("button", { name: "Open chat assistant" }).click();
    await page.getByRole("dialog").waitFor();
    await page.screenshot({ path: path.join(os.tmpdir(), "weedwise-chat-desktop.png") });
    await page.getByRole("button", { name: "Show chat in page" }).click();
    await page.getByRole("dialog").waitFor({ state: "hidden" });
    assert.equal(await page.locator("#chat-messages article").count(), 2);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("button", { name: "Open chat assistant" }).click();
    await page.getByRole("dialog").waitFor();
    assert(await page.getByRole("button", { name: "Send message" }).isVisible());
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    const box = await page.getByRole("dialog").boundingBox();
    assert(box.y >= 0 && box.y + box.height <= 845);
    await page.screenshot({ path: path.join(os.tmpdir(), "weedwise-chat-mobile.png") });
    await page.getByRole("button", { name: "Close chat", exact: true }).click();
    assert.equal(await page.locator("#example-gallery button").count(), 12);
    await page.getByRole("tab", { name: "Examples", exact: true }).click();
    await page.locator('[data-example="maize.jpg"]').click();
    await page.getByText("Photo ready.", { exact: true }).waitFor();
    await page.getByRole("button", { name: "Run identification" }).click();
    await page.getByText("Prediction complete.", { exact: true }).waitFor({ timeout: 60000 });
    assert.equal(await page.locator("#prediction").innerText(), "Maize");
    assert.deepEqual(errors, []);
    console.log("PASS shared floating window, mobile layout and real maize model prediction");
  } finally {
    await browser.close();
    server.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
