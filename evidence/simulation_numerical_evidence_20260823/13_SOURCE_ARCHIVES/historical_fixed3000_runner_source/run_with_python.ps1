param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Script,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = "Stop"
$base = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $base

$candidates = @()
if ($env:PH_CHALLENGE_PYTHON) {
    $candidates += [pscustomobject]@{ Command = $env:PH_CHALLENGE_PYTHON; Prefix = @() }
}
$venvPython = Join-Path $base ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    $candidates += [pscustomobject]@{ Command = $venvPython; Prefix = @() }
}
$candidates += [pscustomobject]@{ Command = "python"; Prefix = @() }
$candidates += [pscustomobject]@{ Command = "py"; Prefix = @("-3.12") }
$candidates += [pscustomobject]@{ Command = "py"; Prefix = @("-3.11") }
$candidates += [pscustomobject]@{ Command = "py"; Prefix = @("-3.10") }
$candidates += [pscustomobject]@{ Command = "py"; Prefix = @("-3") }

$probe = "import sys, numpy, scipy, torch; assert sys.version_info >= (3, 10)"
$selected = $null
foreach ($candidate in $candidates) {
    try {
        & $candidate.Command @($candidate.Prefix) -c $probe 2>$null
        if ($LASTEXITCODE -eq 0) {
            $selected = $candidate
            break
        }
    }
    catch {
        continue
    }
}

if ($null -eq $selected) {
    Write-Host "No usable Python environment was found." -ForegroundColor Red
    Write-Host "Run 00_INSTALL_ENV.cmd once, or set PH_CHALLENGE_PYTHON to a Python 3.10+ executable with numpy/scipy/torch installed."
    exit 1
}

Write-Host "Using Python: $($selected.Command) $($selected.Prefix -join ' ')"
& $selected.Command @($selected.Prefix) $Script @ScriptArgs
exit $LASTEXITCODE
