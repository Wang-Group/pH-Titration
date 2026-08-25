param(
    [ValidateSet("Full", "Quick")]
    [string]$Mode = "Full",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

function Find-BasePython {
    $candidates = @(
        @{ Command = "py"; Arguments = @("-3.11") },
        @{ Command = "py"; Arguments = @("-3.10") },
        @{ Command = "py"; Arguments = @("-3.12") },
        @{ Command = "python"; Arguments = @() }
    )
    foreach ($candidate in $candidates) {
        try {
            & $candidate.Command @($candidate.Arguments) -c "import sys; assert sys.version_info >= (3, 10) and sys.maxsize > 2**32" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
        }
    }
    throw "A 64-bit Python 3.10-3.12 installation was not found. Install 64-bit Python 3.11 and run again."
}

Set-Location $Root
if (-not (Test-Path -LiteralPath $VenvPython)) {
    $base = Find-BasePython
    Write-Host "Creating local Python environment..."
    & $base.Command @($base.Arguments) -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv." }
}

$ImportsOk = $false
try {
    & $VenvPython -c "import numpy, scipy, torch, matplotlib" 2>$null
    $ImportsOk = ($LASTEXITCODE -eq 0)
} catch {
}
if (-not $ImportsOk) {
    Write-Host "Installing required packages. The first PyTorch installation may take several minutes..."
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}

& $VenvPython (Join-Path $Root "check_environment.py")
if ($LASTEXITCODE -ne 0) { throw "Environment or model validation failed." }

if ($Device -eq "cuda") {
    & $VenvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA was requested but PyTorch cannot access a CUDA GPU'"
    if ($LASTEXITCODE -ne 0) { throw "CUDA validation failed. Run the CPU launcher instead." }
}

if ($Mode -eq "Quick") {
    & $VenvPython (Join-Path $Root "direction_assisted_rl_comparison.py") `
        --algorithms ppo a2c reinforce `
        --seeds 101 `
        --train-steps 50 `
        --training-pool-size 20 `
        --eval-tasks 10 `
        --eval-interval 25 `
        --torch-threads 1 `
        --device $Device `
        --output-dir (Join-Path $Root "results_quick_corrected")
} else {
    & $VenvPython (Join-Path $Root "direction_assisted_rl_comparison.py") `
        --algorithms ppo a2c reinforce `
        --seeds 101 202 303 404 555 `
        --train-steps 25000 `
        --training-pool-size 5000 `
        --eval-tasks 1000 `
        --eval-interval 5000 `
        --torch-threads 1 `
        --device $Device `
        --output-dir (Join-Path $Root "results_full")
}
if ($LASTEXITCODE -ne 0) { throw "Experiment failed. Review the final console messages." }

Write-Host "Finished. Open RESULT_SUMMARY.md in the result directory."
