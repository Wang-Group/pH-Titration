param(
    [string]$Notebook = "particle_count_timing_benchmark_20260804.ipynb",
    [string]$ExecutedNotebook = "particle_count_timing_benchmark_executed.ipynb",
    [string]$CompletionMarker = "RUN_COMPLETE.txt"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Results = Join-Path $Root "particle_count_timing_results"
$Log = Join-Path $Root "particle_count_timing_run.log"
Set-Location -LiteralPath $Root

function Log([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
}

function Has-NotebookDependencies([string]$Python) {
    try {
        & $Python -c "import struct,sys,numpy,matplotlib,nbconvert,jupyter; raise SystemExit(0 if sys.version_info >= (3,10) and struct.calcsize('P')*8 == 64 else 1)" *> $null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Is-UsablePython([string]$Python) {
    if (-not $Python -or -not (Test-Path -LiteralPath $Python)) { return $false }
    try {
        & $Python -c "import struct,sys; raise SystemExit(0 if sys.version_info >= (3,10) and struct.calcsize('P')*8 == 64 else 1)" *> $null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Resolve-Python {
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:PH_REVIEW_PYTHON) { $candidates.Add($env:PH_REVIEW_PYTHON) }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($selector in @("-3.12", "-3.11", "-3.10", "-3")) {
            try {
                $resolved = (& $launcher.Source $selector -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1).Trim()
                if ($resolved) { $candidates.Add($resolved) }
            } catch {}
        }
    }
    foreach ($known in @(
        $(if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX "python.exe" }),
        $(if ($env:USERPROFILE) { Join-Path $env:USERPROFILE "anaconda3\python.exe" }),
        $(if ($env:USERPROFILE) { Join-Path $env:USERPROFILE "miniconda3\python.exe" }),
        $(if ($env:ProgramData) { Join-Path $env:ProgramData "anaconda3\python.exe" }),
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )) { if ($known) { $candidates.Add($known) } }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $candidates.Add($pythonCommand.Source) }
    $valid = @($candidates | Select-Object -Unique | Where-Object { Is-UsablePython $_ })
    foreach ($candidate in $valid) {
        if (Has-NotebookDependencies $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
    }
    if ($valid.Count -gt 0) { return (Resolve-Path -LiteralPath $valid[0]).Path }
    throw "No 64-bit Python 3.10+ was found. Set PH_REVIEW_PYTHON or install Python 3.11+."
}

function Invoke-Logged([string]$Python, [string[]]$Arguments) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($exitCode -ne 0) { throw "Command failed with exit code $exitCode" }
}

try {
    New-Item -ItemType Directory -Path $Results -Force | Out-Null
    $failedMarker = Join-Path $Root "RUN_FAILED.txt"
    if (Test-Path -LiteralPath $failedMarker) { [System.IO.File]::Delete($failedMarker) }
    $completionPath = Join-Path $Root $CompletionMarker
    if (Test-Path -LiteralPath $completionPath) { [System.IO.File]::Delete($completionPath) }
    Log "Starting particle-count timing benchmark."
    $Python = Resolve-Python
    if (-not (Has-NotebookDependencies $Python)) {
        $Venv = Join-Path $Root ".venv"
        $VenvPython = Join-Path $Venv "Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $VenvPython)) {
            Log "Creating local virtual environment."
            Invoke-Logged $Python @("-m", "venv", $Venv)
        }
        $Python = $VenvPython
        Log "Installing notebook dependencies."
        Invoke-Logged $Python @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Logged $Python @("-m", "pip", "install", "-r", (Join-Path $Root "requirements.txt"))
    }
    Log "Python: $Python"
    Invoke-Logged $Python @(
        "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", $Notebook,
        "--output", $ExecutedNotebook,
        "--ExecutePreprocessor.timeout=-1"
    )
    Set-Content -LiteralPath (Join-Path $Root $CompletionMarker) -Value (Get-Date -Format "o") -Encoding ASCII
    Log "Benchmark completed. Results: $Results"
}
catch {
    Log ("FAILED: " + $_.Exception.Message)
    Set-Content -LiteralPath (Join-Path $Root "RUN_FAILED.txt") -Value $_.Exception.ToString() -Encoding UTF8
    exit 1
}
