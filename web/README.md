# WeedWise V2 on Vercel

The web app uses the full 48-class classifier with temperature scaling. The browser sends resized pixels to a Vercel Node.js function for ONNX inference. The endpoint returns scores without saving photos. The Examples tab has a selectable gallery of 12 attributed test photos. The public application also includes a searchable species list and evaluation results. Fieldwise's SQLite history and review workflow use the local Streamlit app.

From the repository root, export the verified V2 checkpoint and build the static assets:

```powershell
.\.venv\Scripts\python.exe -m pip install onnx==1.20.1
.\.venv\Scripts\python.exe scripts/export_web_model.py
npm.cmd ci --prefix web
npm.cmd run build --prefix web
```

The export requires the saved V2 checkpoint and release manifest, along with the prepared original Plant Seedlings test images. Generated model assets and build output are ignored by Git. Run `vercel deploy --prod` from `web/dist` using the existing Vercel project. The hosted application credits the original datasets and their usage terms.
