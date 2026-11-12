param(
    [string]$Python = '',
    [string]$Data = '',
    [int[]]$Seeds = @(42, 43, 44),
    [ValidateSet('baseline', 'conv1', 'full')]
    [string[]]$Variants = @('baseline', 'conv1', 'full'),
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]*$')]
    [string]$Tag = 'retrain',
    [ValidateRange(1, 10000)]
    [int]$Epochs = 60,
    [ValidateRange(1, 1024)]
    [int]$Threads = 4,
    [switch]$DownloadData,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
# Keep this file ASCII-only for Windows PowerShell 5.1 encoding compatibility.
# Defaults work in any checkout; explicit relative paths are relative to the caller.
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if ([string]::IsNullOrWhiteSpace($Data)) {
    $Data = Join-Path $PSScriptRoot 'data'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python. Install the environment following REPRODUCE.md, or pass -Python."
}
if ($Tag -eq 'main') {
    throw 'The main tag is reserved for archived paper experiments. Use retrain or another new tag.'
}
$Python = (Resolve-Path -LiteralPath $Python).Path
$Data = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Data)
Write-Host "Python: $Python"
Write-Host "Data: $Data"
Push-Location $PSScriptRoot
try {
    & $Python -c 'import sys, torch, torchvision, numpy; print(sys.executable); print(torch.__version__); print(torchvision.__version__)'
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
    if ($DownloadData -and -not $CheckOnly) {
        & $Python -c 'import sys; from torchvision.datasets import CIFAR10; CIFAR10(sys.argv[1], train=True, download=True); CIFAR10(sys.argv[1], train=False, download=True)' $Data
        if ($LASTEXITCODE -ne 0) { throw 'Data download failed.' }
    }
    & $Python -c 'import sys; from torchvision.datasets import CIFAR10; CIFAR10(sys.argv[1], train=True, download=False); CIFAR10(sys.argv[1], train=False, download=False)' $Data
    if ($LASTEXITCODE -ne 0) { throw 'CIFAR-10 data check failed. Use -DownloadData or pass -Data.' }
    Write-Host 'CIFAR-10 integrity check passed.'
    foreach ($seed in $Seeds) {
        foreach ($variant in $Variants) {
            $runPath = Join-Path $PSScriptRoot "runs\${Tag}_${variant}_s${seed}"
            if ((Test-Path -LiteralPath (Join-Path $runPath 'config.json')) -and
                -not (Test-Path -LiteralPath (Join-Path $runPath 'last.pt'))) {
                throw "Existing run has no resumable checkpoint: $runPath. Use another -Tag."
            }
        }
    }
    if ($CheckOnly) {
        Write-Host 'Environment check passed. No training started.'
        return
    }
    foreach ($seed in $Seeds) {
        foreach ($variant in $Variants) {
            $runPath = Join-Path $PSScriptRoot "runs\${Tag}_${variant}_s${seed}"
            $arguments = @('-u', 'train.py', '--data', $Data, '--variant', $variant,
                           '--seed', $seed, '--epochs', $Epochs, '--batch-size', 128,
                           '--lr', 0.05, '--threads', $Threads, '--tag', $Tag)
            if (Test-Path -LiteralPath (Join-Path $runPath 'last.pt')) { $arguments += '--resume' }
            Write-Host "Starting $variant seed=$seed epochs=$Epochs threads=$Threads"
            Write-Host 'Progress prints after each full epoch (training plus validation).'
            & $Python @arguments
            if ($LASTEXITCODE -ne 0) { throw "Training failed: $variant seed $seed" }
        }
    }
    Write-Host 'Training finished. Run run_evaluation.ps1 with the same Python, Data, Tag, Seeds and Variants.'
} finally { Pop-Location }
