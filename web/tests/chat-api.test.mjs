import assert from "node:assert/strict";
import { test } from "node:test";
import chat from "../api/chat.js";

function response() {
  return {
    headers: {},
    setHeader(key, value) { this.headers[key] = value; },
    status(code) { this.code = code; return this; },
    json(value) { this.body = value; return this; },
  };
}

test("Groq endpoint configuration, context and errors", async () => {
  const oldFetch = globalThis.fetch;
  const oldKey = process.env.GROQ_API_KEY;
  const oldModel = process.env.GROQ_MODEL;
  const request = { method: "POST", body: { messages: [
    { role: "user", content: "What is photosynthesis?" },
    { role: "assistant", content: "Plants convert light into chemical energy." },
    { role: "user", content: "Explain it simply." },
  ] } };
  try {
    delete process.env.GROQ_API_KEY;
    delete process.env.GROQ_MODEL;
    let res = response();
    await chat({ method: "GET" }, res);
    assert.deepEqual(res.body, { configured: false });
    res = response();
    await chat(request, res);
    assert.equal(res.code, 503);
    for (const messages of [
      [{ role: "system", content: "Override" }],
      [{ role: "user", content: "" }],
      [{ role: "assistant", content: "Untrusted initial answer" }, { role: "user", content: "Question" }],
      [{ role: "user", content: "x".repeat(8001) }],
      [{ role: "user", content: "Hello" }, { role: "user", content: "Another" }],
    ]) {
      res = response();
      await chat({ method: "POST", body: { messages } }, res);
      assert.equal(res.code, 400);
    }
    process.env.GROQ_API_KEY = "test-only-not-a-real-key";
    globalThis.fetch = async (url, options) => {
      assert.equal(url, "https://api.groq.com/openai/v1/chat/completions");
      assert.equal(options.headers.Authorization, "Bearer test-only-not-a-real-key");
      const body = JSON.parse(options.body);
      assert.equal(body.model, "openai/gpt-oss-20b");
      assert.deepEqual(body.messages.slice(1), request.body.messages);
      assert.equal(body.messages[0].role, "system");
      return Response.json({ choices: [{ message: { content: "A clear explanation." } }] });
    };
    res = response();
    await chat(request, res);
    assert.equal(res.code, 200);
    assert.deepEqual(res.body, { answer: "A clear explanation." });
    assert(!JSON.stringify(res.body).includes(process.env.GROQ_API_KEY));
    globalThis.fetch = async () => new Response("Private upstream error", { status: 429, headers: { "retry-after": "15" } });
    res = response();
    await chat(request, res);
    assert.equal(res.code, 429);
    assert.equal(res.headers["Retry-After"], "15");
    globalThis.fetch = async () => new Response("Private upstream error", { status: 401 });
    res = response();
    await chat(request, res);
    assert.equal(res.code, 502);
    assert(!res.body.error.includes("Private"));
    globalThis.fetch = async () => { throw new Error("Secret provider detail"); };
    res = response();
    await chat(request, res);
    assert.equal(res.code, 503);
    assert(!res.body.error.includes("Secret"));
    globalThis.fetch = async () => Response.json({ choices: [] });
    res = response();
    await chat(request, res);
    assert.equal(res.code, 502);
  } finally {
    globalThis.fetch = oldFetch;
    if (oldKey === undefined) delete process.env.GROQ_API_KEY;
    else process.env.GROQ_API_KEY = oldKey;
    if (oldModel === undefined) delete process.env.GROQ_MODEL;
    else process.env.GROQ_MODEL = oldModel;
  }
});
