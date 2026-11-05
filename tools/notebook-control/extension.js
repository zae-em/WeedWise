const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
let busy = false;

function projectRoot() {
  return (vscode.workspace.workspaceFolders || []).find(f => f.uri.scheme === 'file' && fs.existsSync(path.join(f.uri.fsPath, '.weed-control', 'enabled')))?.uri.fsPath;
}
function localPath(root, name) {
  const result = path.resolve(root, name);
  const relative = path.relative(root, result);
  if (relative.startsWith('..') || path.isAbsolute(relative)) throw Error('Path outside project');
  return result;
}
function save(root, name, data) {
  fs.writeFileSync(path.join(root, '.weed-control', name), JSON.stringify(data, null, 2));
}
function snapshot(notebook) {
  return notebook.getCells().map(cell => ({
    index: cell.index, execution: cell.executionSummary,
    outputs: cell.outputs.map(output => ({items: output.items.map(item => ({mime: item.mime, text: Buffer.from(item.data).toString('utf8')}))}))
  }));
}

async function runPending() {
  if (busy) return;
  const root = projectRoot();
  if (!root) return;
  const requestPath = path.join(root, '.weed-control', 'request.json');
  if (!fs.existsSync(requestPath)) return;
  const request = JSON.parse(fs.readFileSync(requestPath, 'utf8'));
  const resultPath = path.join(root, '.weed-control', `${request.id}.json`);
  if (fs.existsSync(resultPath)) return;
  busy = true;
  const result = {id: request.id, action: request.action, status: 'running', started: new Date().toISOString()};
  save(root, `${request.id}.json`, result);
  let listener;
  try {
    if (request.action === 'status') {
      const commands = await vscode.commands.getCommands(true);
      result.commands = ['notebook.cell.execute', 'notebook.execute', 'colab.upload', 'colab.mountServer'].filter(c => commands.includes(c));
      result.notebooks = vscode.workspace.notebookDocuments.map(n => ({path: n.uri.fsPath, type: n.notebookType, cells: n.cellCount}));
      result.remoteRoots = (vscode.workspace.workspaceFolders || []).filter(f => f.uri.scheme === 'colab').map(f => f.uri.toString());
    } else if (['execute', 'appendExecute', 'replaceExecute'].includes(request.action)) {
      const uri = vscode.Uri.file(localPath(root, request.notebook));
      if (!uri.fsPath.endsWith('.ipynb') || !path.relative(root, uri.fsPath).startsWith('training')) throw Error('Only project training notebooks may be executed');
      const notebook = await vscode.workspace.openNotebookDocument(uri);
      const editor = await vscode.window.showNotebookDocument(notebook, {preserveFocus: true, preview: false});
      let index = request.index;
      if (request.action === 'appendExecute') {
        index = notebook.cellCount;
        const edit = new vscode.WorkspaceEdit();
        const cell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, request.source, 'python');
        cell.metadata = {weed_action_id: request.id};
        edit.set(uri, [vscode.NotebookEdit.insertCells(index, [cell])]);
        if (!(await vscode.workspace.applyEdit(edit))) throw Error('Could not append project cell');
      }
      if (request.action === 'replaceExecute') {
        if (index < 0 || index >= notebook.cellCount) throw Error('Invalid cell index');
        const edit = new vscode.WorkspaceEdit();
        const cell = new vscode.NotebookCellData(vscode.NotebookCellKind.Code, request.source, 'python');
        cell.metadata = {weed_action_id: request.id};
        edit.set(uri, [vscode.NotebookEdit.replaceCells(new vscode.NotebookRange(index, index + 1), [cell])]);
        if (!(await vscode.workspace.applyEdit(edit))) throw Error('Could not replace dedicated transfer cell');
      }
      if (index < 0 || index >= notebook.cellCount) throw Error('Invalid cell index');
      await notebook.save();
      listener = vscode.workspace.onDidChangeNotebookDocument(event => {
        if (event.notebook.uri.toString() === uri.toString()) {
          save(root, `${request.id}.outputs.json`, snapshot(notebook));
          event.notebook.save();
        }
      });
      editor.selection = new vscode.NotebookRange(index, index + 1);
      await vscode.commands.executeCommand('notebook.cell.execute', {document: uri, ranges: [{start: index, end: index + 1}]});
      // Execution command may return before completion. Observe the actual cell execution summary.
      const deadline = Date.now() + (request.timeoutSeconds || 300) * 1000;
      while (Date.now() < deadline) {
        const cell = notebook.cellAt(index);
        if (cell.executionSummary?.timing?.endTime !== undefined || cell.executionSummary?.success !== undefined) {
          result.execution = cell.executionSummary;
          result.outputs = snapshot(notebook).find(c => c.index === index)?.outputs;
          result.success = cell.executionSummary.success;
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      if (!result.execution) throw Error('Cell did not complete within timeout; inspect VS Code kernel selection or running cell');
      await notebook.save();
    } else if (request.action === 'upload') {
      const uris = request.files.map(file => vscode.Uri.file(localPath(root, file)));
      await vscode.commands.executeCommand('colab.upload', uris[0], uris);
      result.note = 'Upload command returned; verify remote file hashes in a notebook cell before training.';
    } else if (request.action === 'transfer') {
      const remoteRoots = (vscode.workspace.workspaceFolders || []).filter(f => f.uri.scheme === 'colab');
      if (remoteRoots.length !== 1) throw Error('Mount exactly one Colab server for unambiguous transfer');
      result.uploaded = [];
      for (const file of request.files) {
        const source = localPath(root, file);
        const relative = path.relative(root, source);
        if (!['dataset', 'training'].some(folder => relative.startsWith(folder + path.sep))) throw Error('Only project training files can be transferred');
        const destination = remoteRoots[0].uri.with({path: '/content/' + path.basename(source)});
        const bytes = fs.readFileSync(source);
        await vscode.workspace.fs.writeFile(destination, bytes);
        result.uploaded.push({name: path.basename(source), bytes: bytes.length});
        save(root, `${request.id}.json`, result);
      }
    } else if (request.action === 'mount') {
      await vscode.commands.executeCommand('colab.mountServer');
      result.remoteRoots = (vscode.workspace.workspaceFolders || []).filter(f => f.uri.scheme === 'colab').map(f => f.uri.toString());
    } else if (request.action === 'readRemote') {
      const remoteRoots = (vscode.workspace.workspaceFolders || []).filter(f => f.uri.scheme === 'colab');
      if (remoteRoots.length !== 1) throw Error('Mount exactly one Colab server before file retrieval');
      if (!request.remotePath.startsWith('/content/')) throw Error('Only temporary /content project artifacts may be read');
      const uri = remoteRoots[0].uri.with({path: request.remotePath});
      const bytes = await vscode.workspace.fs.readFile(uri);
      const target = localPath(root, request.destination);
      if (!['models', 'evaluation'].some(folder => path.relative(root, target).startsWith(folder + path.sep))) throw Error('Download target must be models/ or evaluation/');
      fs.mkdirSync(path.dirname(target), {recursive: true});
      const temporary = target + '.download';
      fs.writeFileSync(temporary, bytes);
      fs.renameSync(temporary, target);
      result.bytes = bytes.byteLength;
      result.sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
    } else throw Error('Unsupported project action');
    result.status = 'complete';
  } catch (error) {
    result.status = 'error';
    result.error = String(error?.message || error);
  } finally {
    listener?.dispose();
    result.finished = new Date().toISOString();
    save(root, `${request.id}.json`, result);
    busy = false;
  }
}

function activate(context) {
  const root = projectRoot();
  if (root) {
    save(root, 'ready.json', {version: '0.1.2', activated: new Date().toISOString(), workspace: root});
    const requestPath = path.join(root, '.weed-control', 'request.json');
    if (fs.existsSync(requestPath)) {
      const oldRequest = JSON.parse(fs.readFileSync(requestPath, 'utf8'));
      const oldResult = path.join(root, '.weed-control', `${oldRequest.id}.json`);
      if (fs.existsSync(oldResult)) {
        const state = JSON.parse(fs.readFileSync(oldResult, 'utf8'));
        if (state.status === 'running') save(root, `${oldRequest.id}.json`, {...state, status: 'interrupted', note: 'Extension host restarted; requeue explicitly. A remote training worker, if started, is checked before resumption.'});
      }
    }
  }
  context.subscriptions.push(vscode.commands.registerCommand('weed.runPending', runPending));
  const interval = setInterval(() => runPending().catch(error => {
    const current = projectRoot();
    if (current) save(current, 'bridge_error.json', {error: String(error)});
  }), 1500);
  context.subscriptions.push({dispose: () => clearInterval(interval)});
  runPending();
}
module.exports = {activate};
