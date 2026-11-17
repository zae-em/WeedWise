const el = id => document.getElementById(id);
const PREFIX = "weedwise.chat.v1.";
const pending = new Set();
const errors = new Map();
const conversations = new Map();
let activeId = null;
let configured = false;

function validConversation(value) {
  return value && typeof value.id === "string" && typeof value.title === "string" &&
    Number.isFinite(value.updated) && Array.isArray(value.messages) &&
    value.messages.every((message, index) => message && typeof message.content === "string" &&
      message.role === (index % 2 ? "assistant" : "user"));
}

function loadHistory() {
  try {
    for (let index = 0; index < localStorage.length; index++) {
      const key = localStorage.key(index);
      if (!key.startsWith(PREFIX)) continue;
      try {
        const value = JSON.parse(localStorage.getItem(key));
        if (validConversation(value) && key === PREFIX + value.id) conversations.set(value.id, value);
      } catch {
        // A damaged record does not prevent loading the remaining history.
      }
    }
  } catch {
    el("chat-storage").textContent = "Browser storage is unavailable. Conversations will last for this page session.";
  }
}

function save(conversation) {
  conversation.updated = Date.now();
  try {
    localStorage.setItem(PREFIX + conversation.id, JSON.stringify(conversation));
    el("chat-storage").textContent = "";
  } catch {
    el("chat-storage").textContent = "This conversation could not be saved in your browser. Keep the page open to retain it.";
  }
}

function current() {
  return conversations.get(activeId);
}

function renderHistory() {
  const sorted = [...conversations.values()].sort((a, b) => b.updated - a.updated);
  el("history-empty").hidden = sorted.length > 0;
  el("chat-history").replaceChildren(...sorted.map(conversation => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = conversation.title;
    button.title = conversation.title;
    button.setAttribute("aria-current", String(conversation.id === activeId));
    button.addEventListener("click", () => {
      activeId = conversation.id;
      el("chat-input").value = "";
      render();
    });
    item.append(button);
    return item;
  }));
}

function render() {
  const conversation = current();
  const waiting = pending.has(activeId);
  const unanswered = conversation?.messages.at(-1)?.role === "user";
  const messages = conversation?.messages || [];
  el("conversation-title").textContent = conversation?.title || "New conversation";
  el("chat-messages").replaceChildren(...messages.map(message => {
    const article = document.createElement("article");
    article.className = "chat-message chat-message-" + message.role;
    const label = document.createElement("strong");
    const text = document.createElement("div");
    label.textContent = message.role === "user" ? "You" : "Assistant";
    text.textContent = message.content;
    article.append(label, text);
    return article;
  }));
  if (!messages.length) {
    const welcome = document.createElement("div");
    welcome.className = "chat-welcome";
    const heading = document.createElement("strong");
    heading.textContent = "What would you like to know?";
    const text = document.createElement("p");
    text.textContent = "Ask about plants, science, everyday topics or something you are learning.";
    welcome.append(heading, text);
    el("chat-messages").append(welcome);
  }
  el("chat-error").textContent = errors.get(activeId) || "";
  el("chat-error").hidden = !errors.has(activeId);
  el("chat-retry").hidden = !unanswered || waiting || !configured;
  el("chat-send").disabled = !configured || waiting || unanswered || !el("chat-input").value.trim();
  el("chat-input").disabled = waiting || Boolean(unanswered);
  el("chat-request-status").textContent = waiting ? "Assistant is replying…" : unanswered ? "Your last message has no reply yet." : "";
  renderHistory();
  el("chat-messages").scrollTop = el("chat-messages").scrollHeight;
}

function contextMessages(messages) {
  let start = messages.length - 1;
  let characters = messages[start].content.length;
  while (start >= 2 && messages.length - start + 2 <= 31) {
    const pairSize = messages[start - 1].content.length + messages[start - 2].content.length;
    if (characters + pairSize > 18000 || messages[start - 1].content.length > 8000) break;
    characters += pairSize;
    start -= 2;
  }
  return messages.slice(start).map(({ role, content }) => ({ role, content }));
}

async function requestReply(conversation) {
  const id = conversation.id;
  if (pending.has(id)) return;
  pending.add(id);
  errors.delete(id);
  render();
  try {
    const response = await fetch("./api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: contextMessages(conversation.messages) }),
      signal: AbortSignal.timeout(28000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The assistant could not reply. Please retry.");
    if (typeof result.answer !== "string" || !result.answer.trim()) throw new Error("The assistant returned an empty reply.");
    conversation.messages.push({ role: "assistant", content: result.answer });
    save(conversation);
  } catch (error) {
    const text = ["TimeoutError", "AbortError"].includes(error.name)
      ? "The reply took too long. Please retry your last message."
      : error instanceof TypeError ? "The chat service could not be reached. Check your connection and retry." : error.message;
    errors.set(id, text);
  } finally {
    pending.delete(id);
    render();
  }
}

el("chat-form").addEventListener("submit", event => {
  event.preventDefault();
  const content = el("chat-input").value.trim();
  if (!configured || !content || content.length > 2000 || pending.has(activeId) ||
      current()?.messages.at(-1)?.role === "user") return;
  let conversation = current();
  if (!conversation) {
    conversation = { id: crypto.randomUUID(), title: content.slice(0, 65), updated: Date.now(), messages: [] };
    activeId = conversation.id;
    conversations.set(activeId, conversation);
  }
  conversation.messages.push({ role: "user", content });
  save(conversation);
  el("chat-input").value = "";
  requestReply(conversation);
});
el("chat-input").addEventListener("input", () => {
  el("chat-send").disabled = !configured || pending.has(activeId) ||
    current()?.messages.at(-1)?.role === "user" || !el("chat-input").value.trim();
});
el("chat-input").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    el("chat-form").requestSubmit();
  }
});
el("chat-retry").addEventListener("click", () => {
  if (configured && current()?.messages.at(-1)?.role === "user") requestReply(current());
});
el("new-chat").addEventListener("click", () => {
  activeId = null;
  el("chat-input").value = "";
  render();
  el("chat-input").focus();
});

const dialog = el("chat-dialog");
function restoreWorkspace() {
  el("chat-home").append(el("chat-workspace"));
  el("chat-away").hidden = true;
  el("chat-launcher").hidden = false;
}
function showInPage() {
  if (dialog.open) dialog.close();
  restoreWorkspace();
  location.hash = "chat";
  el("chat").scrollIntoView();
  el("chat-input").focus();
}
el("chat-launcher").addEventListener("click", () => {
  el("chat-dialog-body").append(el("chat-workspace"));
  el("chat-away").hidden = false;
  el("chat-launcher").hidden = true;
  dialog.showModal();
  el("chat-input").focus();
});
el("chat-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("close", restoreWorkspace);
el("chat-fullpage").addEventListener("click", showInPage);
el("chat-return").addEventListener("click", showInPage);
window.addEventListener("hashchange", () => {
  if (location.hash === "#chat" && dialog.open) dialog.close();
});
window.addEventListener("storage", event => {
  if (!event.key?.startsWith(PREFIX) || !event.newValue) return;
  try {
    const conversation = JSON.parse(event.newValue);
    if (validConversation(conversation) && !pending.has(conversation.id)) {
      conversations.set(conversation.id, conversation);
      render();
    }
  } catch {
    // Ignore incomplete writes from another browser tab.
  }
});

loadHistory();
activeId = [...conversations.values()].sort((a, b) => b.updated - a.updated)[0]?.id || null;
render();
try {
  const response = await fetch("./api/chat", { cache: "no-store" });
  if (!response.ok) throw new Error("Chat availability could not be checked. Refresh to retry.");
  const status = await response.json();
  configured = status.configured === true;
  el("chat-configuration").textContent = configured
    ? "Ready for your questions."
    : "Chat is ready for setup. Replies will be available when the site owner adds the Groq API key.";
} catch {
  el("chat-configuration").textContent = "Chat availability could not be checked. Refresh to retry.";
}
render();
