param(
    [string]$Python = '',
    [string]$Data = '',
    [int]$Threads = 4
)
$ErrorActionPreference = 'Stop'
# ASCII source for Windows PowerShell 5.1; Unicode paths come from the filesystem.
$workspacePath = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $workspacePath 'fasternet-course-restart\.venv\Scripts\python.exe'
}
if ([string]::IsNullOrWhiteSpace($Data)) {
    $Data = Join-Path $workspacePath 'fasternet-reproduction\data'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Data -PathType Container)) { throw "Data directory not found: $Data" }
Push-Location $PSScriptRoot
try {
    & $Python -u final_evaluation.py --data $Data --threads $Threads
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed. Check logs/eval_*.log.' }
    & $Python -u summarize_results.py
    if ($LASTEXITCODE -ne 0) { throw 'Result verification or summary failed.' }
    Write-Host 'Done. Open results/RESULTS_SUMMARY.md for the final tables.'
} finally { Pop-Location }
