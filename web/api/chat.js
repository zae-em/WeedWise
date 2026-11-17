const DEFAULT_MODEL = "openai/gpt-oss-20b";
const SYSTEM = "You are the WeedWise chat assistant. Answer general questions clearly and helpfully. " +
  "Use the user's language. Ask for clarification when needed and acknowledge uncertainty. " +
  "You can discuss plants and agriculture as well as other topics. " +
  "You have no access to photos, classifier predictions, private files, or live web search in this chat.";

export default async function chat(req, res) {
  res.setHeader("Cache-Control", "no-store");
  const key = process.env.GROQ_API_KEY?.trim();
  if (req.method === "GET") return res.status(200).json({ configured: Boolean(key) });
  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "Use POST to send a chat message." });
  }
  const messages = req.body?.messages;
  if (!Array.isArray(messages) || !messages.length || messages.length > 32 ||
      messages.some(message => !message || !["user", "assistant"].includes(message.role) ||
        typeof message.content !== "string" || !message.content.trim() || message.content.length > 8000) ||
      messages.reduce((sum, message) => sum + message.content.length, 0) > 18000 ||
      messages[0].role !== "user" || messages.at(-1).role !== "user" ||
      messages.some((message, index) => message.role !== (index % 2 ? "assistant" : "user"))) {
    return res.status(400).json({ error: "Send a valid conversation ending with your question." });
  }
  if (!key) {
    return res.status(503).json({ error: "Chat will be available when the Groq API key is configured." });
  }
  try {
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: { "Authorization": "Bearer " + key, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: process.env.GROQ_MODEL?.trim() || DEFAULT_MODEL,
        messages: [{ role: "system", content: SYSTEM }, ...messages.map(({ role, content }) => ({ role, content }))],
        max_completion_tokens: 1200,
      }),
      signal: AbortSignal.timeout(23000),
    });
    if (response.status === 429) {
      const retry = response.headers.get("retry-after");
      if (retry && /^\d{1,6}$/.test(retry)) res.setHeader("Retry-After", retry);
      return res.status(429).json({ error: "Groq's request limit has been reached. Please try again shortly." });
    }
    if (!response.ok) {
      const error = [401, 403].includes(response.status)
        ? "The chat service credentials need attention. Please contact the site owner."
        : "Groq could not answer this request. Please try again shortly.";
      return res.status(502).json({ error });
    }
    const result = await response.json();
    const answer = result.choices?.[0]?.message?.content;
    if (typeof answer !== "string" || !answer.trim() || answer.length > 20000) {
      return res.status(502).json({ error: "Groq returned an empty or invalid answer. Please try again." });
    }
    return res.status(200).json({ answer: answer.trim() });
  } catch {
    return res.status(503).json({ error: "The chat service is taking too long or is unavailable. Please try again." });
  }
}
