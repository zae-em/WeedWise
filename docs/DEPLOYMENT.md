# Deployment readiness and local operation

The verified target is the Windows laptop with Python 3.12, CPU inference and Streamlit bound to
`127.0.0.1:8501`. Start with the README command. The app uses `models/mobilenet_v2/model.pt` and stores SQLite
records under `app/data/`. Permanent storage stays local.

Pinned Python requirements, pinned Node browser-test dependencies, application services, test runners, model
metadata, recovery checkpoints and measured reports support reproducing the local deployment. The Windows
requirements file is not a cross-platform Docker lockfile.

No public site, cloud host or permanent cloud database has been created. A shared/public deployment needs an
explicit access/privacy design, HTTPS, monitoring, independent field validation and local agronomic review.
Those production requirements are outside the verified local deployment and are not claimed complete.

Stop the server with Ctrl+C in its launching terminal or by stopping that app Python process. Back up source,
models, reports, knowledge, saved images and SQLite together. Stop the app before copying SQLite files, or use
SQLite's backup API for a consistent live backup.
