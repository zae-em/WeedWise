# WeedWise V2 on Vercel

## Install as a mobile app

Open https://weedwise-v2-fyp.vercel.app/ and use **Install WeedWise** near the bottom of the page. Supporting browsers show an install prompt. Android users can also choose **Install app** or **Add to Home screen** from Chrome's menu. On iPhone/iPad, open the link in Safari and use **Share → Add to Home Screen**, enabling **Open as Web App** if shown.

The installed PWA uses the same routes and server APIs, opens in a standalone window, and has the WeedWise icon. Identification and chat require internet access. Only the offline notice page and its logo are cached; photos, model predictions, chat messages and API responses are not cached by the service worker. Existing chat history continues to use browser-local storage; separate browsers or installation contexts may have separate history. The service worker has a build-generated cache version and waits for existing app windows to close before activating an update.

## Chat assistant

The Chat navigation link and floating bottom-right button open the same conversation interface. Each conversation is saved in this browser's local storage and can be reopened from Previous chats. New chat starts a separate conversation. The recent conversation context is sent to Groq for general questions. History is local to the browser and does not sync between devices. Storage limits or blocked browser storage are reported in the interface.

Set `GROQ_API_KEY` in the Vercel project's server environment variables and redeploy to enable replies. The optional `GROQ_MODEL` variable overrides `openai/gpt-oss-20b`. The key is read only by `api/chat.js`. The browser receives availability status and replies. Groq quotas are shared by all site users and depend on the account's plan. Without a key the interface displays its setup status.

Failed requests keep the user's message in its conversation and offer Retry last message. New conversations and history remain usable while a reply is pending. The endpoint limits context to 32 messages and 18,000 characters. Older messages stay in browser history while recent complete turns are selected for context.

The web app uses the full 48-class classifier with temperature scaling. The browser sends resized pixels to a Vercel Node.js function for ONNX inference. The endpoint returns scores without saving photos. The Examples tab has a selectable gallery of 12 attributed test photos. The public application also includes a searchable species list and evaluation results. Fieldwise's SQLite history and review workflow use the local Streamlit app.

From the repository root, export the verified V2 checkpoint and build the static assets:

```powershell
.\.venv\Scripts\python.exe -m pip install onnx==1.20.1
.\.venv\Scripts\python.exe scripts/export_web_model.py
npm.cmd ci --prefix web
npm.cmd run build --prefix web
```

The export requires the saved V2 checkpoint and release manifest, along with the prepared original Plant Seedlings test images. Generated model assets and build output are ignored by Git. Run `vercel deploy --prod` from `web/dist` using the existing Vercel project. The hosted application credits the original datasets and their usage terms.
