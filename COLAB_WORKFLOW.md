# Historical Colab setup workflow

This setup-stage guide is retained for connection troubleshooting. The completed project now uses
[docs/TRAINING.md](docs/TRAINING.md) and `training/complete_training.ipynb` for full runs. Supported
automatic execution and file retrieval have been implemented. Full Tesla T4 training has finished and the
genuine model is saved locally; see [FINAL_PROJECT_STATUS.md](FINAL_PROJECT_STATUS.md). Training is not
needed to use the local app. The older notebook and START_TRAINING instructions below are historical.

## Connect and verify

1. Open local `training/colab_connection_test.ipynb` in VS Code.
2. Select **Select Kernel > Select Another Kernel > Colab > New Colab Server**.
   Choose a GPU/T4 option if offered. Do not select a paid option without intending to pay.
   **Auto Connect** may choose a CPU machine.
3. Complete Google's sign-in and consent in your own browser when prompted, then return to VS Code.
   No password, verification code or token belongs in chat or project files.
4. If Colab says **Unable to assign server. You have too many, remove one to continue**, use
   **Ctrl+Shift+P > Colab: Remove Server**, select an unused server, and retry. An active server may contain
   work: decide which one can be released yourself.
5. Select the test code cell, press **Shift+Enter**, and save with **Ctrl+S**. A successful test prints
   `cuda_available: true`, the actual GPU name, and `cuda_tensor_test: true`.
   CPU output or notebook GPU metadata alone is not evidence of GPU access.

Your Google account controls availability and quota. T4 access is requested, not guaranteed.
No dataset is needed for this test. If VS Code needs a reload to expose Colab, run **Developer: Reload Window**,
then reopen the local notebook. Local `.venv` is for Windows CPU execution and is not the Colab kernel.

## Upload only when training is ready

1. Run locally:

   ```powershell
   .\.venv\Scripts\python.exe scripts/prepare_colab_bundle.py
   ```

2. In VS Code Explorer, right-click `training/colab_source.zip` and choose **Upload to Colab**.
   Later, do the same for your dataset ZIP under `dataset/`, plus recovery checkpoints if resuming.
   The installed Colab extension exposes this command; this does not prove any transfer has occurred.
3. Open local `training/train_colab.ipynb`, choose the same Colab kernel and run preparation cells one at a time.
   Files uploaded by the extension appear under temporary `/content`. The notebook checks paths before use.
4. Install `requirements-colab.txt` in the notebook. Preserve Colab's preinstalled matched CUDA PyTorch pair;
   never run the Windows `requirements.txt` there. The notebook records actual runtime library versions.
5. Set `DATASET_ZIP` and `DATA_DIR` to the uploaded ZIP and actual extracted class-folder root.
   Safe extraction checks uncompressed size plus a 1 GiB free-space reserve. The uploaded ZIP, extracted
   images, checkpoints and output archives all consume runtime storage; free space is checked again at extraction.
   Remove an uploaded ZIP only after you have confirmed extraction and retained your local original.
6. Inspect the detected classes and split counts. Investigate wrong folder nesting, corrupt images or label issues
   before training. No species names or expected dataset statistics have been assumed.
7. Only when ready, set `START_TRAINING = True` in the training cell and execute that cell.
   ImageNet weights are fetched on this first real training run, not during setup.

## Recover outputs to the laptop

Every completed epoch writes `best.pt` when validation improves and atomically replaces `last.pt`.
These files live only in the runtime until downloaded. An atomic replace protects completed files against
a partial write; it cannot protect them if the whole runtime is deleted.

Use the VS Code **Colab** activity bar **Contents** view and browse `/content/weed_outputs/models/`.
Right-click `last.pt` and `best.pt` > **Download...** and save them to local `models/`, preferably in a run folder.
Download snapshots after useful epochs. Downloads while training is writing may race with file replacement;
the simplest consistent snapshot is taken while the training cell is stopped between runs or after completion.
`last.pt` embeds the best weights too, so one completed last.pt snapshot is sufficient for recovery.

After training finishes, the final notebook cell packages `/content/weed_outputs` into `/content/weed_artifacts.zip`.
Download it through **Colab Contents > weed_artifacts.zip > Download...** to local storage, then run:

```powershell
.\.venv\Scripts\python.exe scripts/import_artifacts.py 'path\to\weed_artifacts.zip'
```

This places bundle `models/` entries under local `models/` and evaluation entries under local `evaluation/`.
It rejects path traversal and refuses to overwrite existing output files; move a previous run into a local
run folder first if there is a collision. It does not execute downloaded checkpoints.

## Session interruption

For a disconnected notebook, reconnect to the same assigned server first; its files may still be present.
For a deleted or expired runtime, upload the source bundle and dataset again. Upload the locally downloaded
`last.pt` (optionally best.pt too); set `config["resume"]` to last.pt's runtime path in the config cell.
Re-run preparation and inspection, then explicitly enable the training cell. Resume restarts after the saved
epoch. Work since the last locally downloaded checkpoint can be lost.

There is no always-on automatic laptop-to-Colab backup service in this project. File transfer uses the official
extension and deliberate Upload/Download actions. Do not rely on runtime storage as a permanent copy.

## Execution capability and fallback

The Colab extension supports notebook execution through VS Code's Jupyter UI. Its installed command manifest
exposes uploads/downloads and server controls. A VS Code command declaration is not a standalone authenticated
Python API or a CLI for this assistant to execute cells in your currently selected cloud kernel.
This agent session has shell access and can prepare notebooks and run local kernels, but has no exposed tool
for controlling the current VS Code Jupyter kernel. Cloud execution is verified only from saved test output.
Google OAuth secrets are not extracted from VS Code to work around this boundary.

If the extension's transfer UI fails, the browser Colab Files pane supports upload/download. Browser Colab
can open a local notebook via File > Upload notebook; explicitly download notebook changes and outputs back
to your laptop and do not choose Save a copy in Drive. The primary prepared workflow is the local VS Code
notebook with official extension transfer, which avoids reliance on a Drive notebook copy.

Official references:

- [Colab VS Code user guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)
- [Extension commands and configuration](https://github.com/googlecolab/colab-vscode/blob/main/package.json)
- [Known browser-only library limitations](https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds)
- [Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits)

## Expansion recovery

For the expanded model, use the existing T4 and the temporary Colab scratchpad with training/browser_entry.py. The equivalent local notebook is training/expanded_training_browser.ipynb. Upload training/expanded_colab_inputs.zip through the Files pane to temporary session storage. The entry code verifies its SHA-256 before starting the worker. Use the runtime status and epoch logs to confirm training. Save scratchpad edits by downloading a local notebook copy. Colab's notebook upload created an incidental Drive notebook shell during the previous attempt, so use the scratchpad for this workflow.

Retrieve the recovery checkpoints, expanded_artifacts.zip and expansion_transfer.json locally. Import with scripts/import_expanded_artifacts.py and run scripts/release_expanded_model.py before switching the app to v2. See RESUME_STATUS.md for the last verified state.
