$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskProjectRoot
& (Join-Path $taskProjectRoot '.venv\Scripts\python.exe') -m streamlit run app/main.py
