param(
    [string]$Python = '',
    [string]$Data = '',
    [int[]]$Seeds = @(42),
    [int]$Epochs = 60,
    [int]$Threads = 4,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
# Keep this file ASCII-only for Windows PowerShell 5.1 encoding compatibility.
# Derive Unicode workspace paths from PowerShell instead of source literals.
$workspacePath = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = Join-Path $workspacePath 'fasternet-course-restart\.venv\Scripts\python.exe'
}
if ([string]::IsNullOrWhiteSpace($Data)) {
    $Data = Join-Path $workspacePath 'fasternet-reproduction\data'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python. Pass -Python with an existing executable."
}
if (-not (Test-Path -LiteralPath (Join-Path $Data 'cifar-10-batches-py') -PathType Container)) {
    throw "CIFAR-10 data not found: $Data. Pass -Data with the dataset parent directory."
}
$Python = (Resolve-Path -LiteralPath $Python).Path
$Data = (Resolve-Path -LiteralPath $Data).Path
Write-Host "Python: $Python"
Write-Host "Data: $Data"
Push-Location $PSScriptRoot
try {
    if ($CheckOnly) {
        & $Python -c 'import sys, torch, torchvision; print(sys.executable); print(torch.__version__); print(torchvision.__version__)'
        if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
        Write-Host 'Environment check passed. No training started.'
        return
    }
    foreach ($seed in $Seeds) {
        foreach ($variant in @('baseline', 'conv1', 'full')) {
            $runPath = Join-Path $PSScriptRoot "runs\main_${variant}_s${seed}"
            $arguments = @('-u', 'train.py', '--data', $Data, '--variant', $variant,
                           '--seed', $seed, '--epochs', $Epochs, '--threads', $Threads, '--tag', 'main')
            if (Test-Path -LiteralPath (Join-Path $runPath 'last.pt')) { $arguments += '--resume' }
            Write-Host "Starting $variant seed=$seed epochs=$Epochs threads=$Threads"
            Write-Host 'Progress prints after each full epoch (training plus validation).'
            & $Python @arguments
            if ($LASTEXITCODE -ne 0) { throw "Training failed: $variant seed $seed" }
        }
    }
    Write-Host 'Training finished. Fix the final protocol before running test evaluation.'
} finally { Pop-Location }
