"""Package a small local extension without npm dependencies or external services."""
import zipfile
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source=ROOT/'tools/notebook-control'
version=json.loads((source/'package.json').read_text())['version']
target=source/f'weed-notebook-control-{version}.vsix'
manifest='''<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
<Metadata><Identity Language="en-US" Id="weed-notebook-control" Version="0.1.0" Publisher="local-weed-project"/><DisplayName>Weed Project Notebook Control</DisplayName><Description xml:space="preserve">Project notebook execution using supported VS Code APIs.</Description><Tags>notebooks</Tags><Categories>Other</Categories><GalleryFlags>Public</GalleryFlags><Properties><Property Id="Microsoft.VisualStudio.Code.Engine" Value="^1.99.0"/><Property Id="Microsoft.VisualStudio.Code.ExtensionDependencies" Value=""/><Property Id="Microsoft.VisualStudio.Code.ExtensionPack" Value=""/><Property Id="Microsoft.VisualStudio.Code.ExtensionKind" Value="workspace"/></Properties></Metadata>
<Installation><InstallationTarget Id="Microsoft.VisualStudio.Code"/></Installation><Dependencies/><Assets><Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true"/></Assets></PackageManifest>'''
content='''<?xml version="1.0" encoding="utf-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="json" ContentType="application/json"/><Default Extension="js" ContentType="application/javascript"/><Default Extension="vsixmanifest" ContentType="text/xml"/></Types>'''
manifest=manifest.replace('Version="0.1.0"',f'Version="{version}"')
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
    bundle.writestr('extension.vsixmanifest',manifest)
    bundle.writestr('[Content_Types].xml',content)
    for name in ['package.json','extension.js']: bundle.write(source/name,'extension/'+name)
control=ROOT/'.weed-control';control.mkdir(exist_ok=True)
(control/'enabled').write_text('Only this project authorizes the local notebook workflow controller.\n')
print(target)
