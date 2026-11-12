param(
    [string]$Python = '',
    [string]$Data = '',
    [int[]]$Seeds = @(42, 43, 44),
    [ValidateSet('baseline', 'conv1', 'full')]
    [string[]]$Variants = @('baseline', 'conv1', 'full'),
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]*$')]
    [string]$Tag = 'retrain',
    [ValidateRange(1, 1024)]
    [int]$Threads = 4,
    [ValidateRange(1, 10000)]
    [int]$Repeats = 30,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
# ASCII source for Windows PowerShell 5.1; Unicode paths come from the filesystem.
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if ([string]::IsNullOrWhiteSpace($Data)) {
    $Data = Join-Path $PSScriptRoot 'data'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if ($Tag -eq 'main') {
    throw 'The main tag is reserved for archived paper experiments. Use your retraining tag.'
}
$Python = (Resolve-Path -LiteralPath $Python).Path
$Data = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Data)
Push-Location $PSScriptRoot
try {
    & $Python -c 'import sys; from torchvision.datasets import CIFAR10; CIFAR10(sys.argv[1], train=False, download=False)' $Data
    if ($LASTEXITCODE -ne 0) { throw 'Environment or data check failed. See REPRODUCE.md.' }
    Write-Host 'CIFAR-10 test data integrity check passed.'
    # Check every requested run before starting expensive evaluation.
    foreach ($seed in $Seeds) {
        foreach ($variant in $Variants) {
            $runName = "${Tag}_${variant}_s${seed}"
            $runPath = Join-Path $PSScriptRoot "runs\$runName"
            $statusPath = Join-Path $runPath 'status.json'
            if (-not (Test-Path -LiteralPath (Join-Path $runPath 'best.pt') -PathType Leaf) -or
                -not (Test-Path -LiteralPath $statusPath -PathType Leaf)) {
                throw "Completed training checkpoint missing: $runName. Run run_all.ps1 first."
            }
            $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
            if ($status.complete -ne $true) { throw "Training is incomplete: $runName" }
            if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "results\$runName")) {
                throw "Results already exist: $runName. Preserve them and use a new training tag for a new experiment."
            }
        }
    }
    if ($CheckOnly) {
        Write-Host 'Evaluation preflight passed. No evaluation started.'
        return
    }
    foreach ($seed in $Seeds) {
        foreach ($variant in $Variants) {
            $runName = "${Tag}_${variant}_s${seed}"
            $checkpoint = Join-Path $PSScriptRoot "runs\$runName\best.pt"
            & $Python -u evaluate_and_export.py --checkpoint $checkpoint --data $Data --threads $Threads --repeats $Repeats
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $runName. Preserve partial results before retrying." }
        }
    }
    Write-Host "Done. New metrics, deployment weights and predictions are in results/${Tag}_*/."
    Write-Host 'Original paper summary tables are preserved; compare your new metrics.json files.'
} finally { Pop-Location }
