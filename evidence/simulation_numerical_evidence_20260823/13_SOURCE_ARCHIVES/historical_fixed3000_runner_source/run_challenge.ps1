param(
    [ValidateSet('Check', 'Quick', 'Screen', 'Full')]
    [string]$Mode = 'Screen',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root
$script:LockFile = Join-Path $Root ('.run_{0}.lock' -f $Mode.ToLowerInvariant())
$script:LockHandle = $null

try {
    $script:LockHandle = [System.IO.File]::Open(
        $script:LockFile,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    Write-Host "Another $Mode run is already using this package. Wait for it to finish before starting again."
    exit 1
}

function Test-Python([string]$Executable) {
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable)) { return $false }
    try {
        & $Executable -c "import struct,sys; raise SystemExit(0 if sys.version_info >= (3,10) and struct.calcsize('P')*8 == 64 else 1)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch { return $false }
}

function Test-Dependencies([string]$Executable) {
    try {
        & $Executable -c "import numpy,scipy,torch,matplotlib" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch { return $false }
}

function Resolve-Python {
    if ($env:PH_REVIEW_PYTHON -and (Test-Python $env:PH_REVIEW_PYTHON)) {
        return (Resolve-Path -LiteralPath $env:PH_REVIEW_PYTHON).Path
    }
    $candidates = New-Object System.Collections.Generic.List[string]
    foreach ($known in @(
        $(if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX 'python.exe' }),
        $(if ($env:USERPROFILE) { Join-Path $env:USERPROFILE 'anaconda3\python.exe' }),
        $(if ($env:USERPROFILE) { Join-Path $env:USERPROFILE 'miniconda3\python.exe' }),
        'D:\Anaconda3\python.exe',
        'D:\anaconda\python.exe',
        'C:\Python312\python.exe',
        'C:\Python311\python.exe',
        'C:\Python310\python.exe'
    )) {
        if ($known) { $candidates.Add($known) }
    }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) { $candidates.Add($pythonCommand.Source) }
    $valid = @($candidates | Select-Object -Unique | Where-Object { Test-Python $_ })
    foreach ($candidate in $valid) {
        if (Test-Dependencies $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
    }
    if ($valid.Count -gt 0) { return (Resolve-Path -LiteralPath $valid[0]).Path }
    throw 'No 64-bit Python 3.10+ was found. Install Python 3.11 or set PH_REVIEW_PYTHON.'
}

function Invoke-LiveCommand([string]$Executable, [string[]]$Arguments) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Executable @Arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            Add-Content -LiteralPath $script:LogFile -Value $line -Encoding UTF8
        }
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $oldPreference }
    if ($exitCode -ne 0) { throw "Command failed with exit code $exitCode`: $Executable $($Arguments -join ' ')" }
}

function Write-RunLog([string]$Message) {
    $line = ('[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message)
    Write-Host $line
    Add-Content -LiteralPath $script:LogFile -Value $line -Encoding UTF8
    Set-Content -LiteralPath $script:ProgressFile -Value $line -Encoding UTF8
}

function Invoke-Step([string]$Name, [string]$ScriptName, [string[]]$Arguments) {
    $marker = Join-Path $script:RunDir ('.done_' + $Name)
    if ((Test-Path -LiteralPath $marker) -and -not $Force) {
        Write-RunLog "SKIP $Name (completion marker exists)."
        return
    }
    Write-RunLog "START $Name"
    $started = Get-Date
    Invoke-LiveCommand $script:PythonExe (@((Join-Path $Root $ScriptName)) + $Arguments)
    Set-Content -LiteralPath $marker -Value (Get-Date -Format 'o') -Encoding ASCII
    Write-RunLog ("DONE {0} in {1}" -f $Name, ((Get-Date) - $started))
}

try {
    $script:PythonExe = Resolve-Python
    if (-not (Test-Dependencies $script:PythonExe)) {
        $venv = Join-Path $Root '.venv'
        $venvPython = Join-Path $venv 'Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $venvPython)) {
            & $script:PythonExe -m venv $venv
        }
        $script:PythonExe = $venvPython
        & $script:PythonExe -m pip install --upgrade pip
        & $script:PythonExe -m pip install -r (Join-Path $Root 'requirements.txt')
    }

    $runName = $Mode.ToLowerInvariant()
    $script:RunDir = Join-Path $Root ("results_{0}" -f $runName)
    New-Item -ItemType Directory -Path $script:RunDir -Force | Out-Null
    $script:LogFile = Join-Path $script:RunDir 'run_challenge.log'
    $script:ProgressFile = Join-Path $script:RunDir 'PROGRESS.txt'
    Write-RunLog "Starting RL-versus-Bayesian challenge: $Mode"
    $cuda = (& $script:PythonExe -c "import torch;print('true' if torch.cuda.is_available() else 'false')" | Select-Object -Last 1).Trim()
    $trainDevice = if ($cuda -eq 'true') { 'cuda' } else { 'cpu' }
    $workers = [Math]::Max(1, [Math]::Min(5, [Environment]::ProcessorCount))
    Write-RunLog "Python: $script:PythonExe; training device: $trainDevice; evaluation workers: $workers"

    Invoke-LiveCommand $script:PythonExe @(
        (Join-Path $Root 'validate_package.py'),
        '--output', (Join-Path $script:RunDir 'package_validation.json')
    )
    if ($Mode -eq 'Check') {
        Set-Content -LiteralPath (Join-Path $script:RunDir 'RUN_COMPLETE_CHECK.txt') -Value (Get-Date -Format 'o') -Encoding ASCII
        Write-RunLog 'Environment and package check completed.'
        exit 0
    }

    if ($Mode -eq 'Quick') {
        $trainSeeds = @('101')
        $evalSeeds = @('9101')
        $trainSteps = '250'
        $poolSize = '60'
        $evalInterval = '125'
        $evalTasks = '2'
        $nominalTasks = '3'
        $stressTasks = '1'
        $nominalParticles = '20'
        $stressParticles = '20'
        $bootstrap = '100'
        $scenarioNames = @('nominal', 'noise_005', 'low_conc_noise')
        $workers = 1
        $ppoBatch = '64'
        $ppoEpochs = '1'
        $sacWarmup = '20'
        $sacBatch = '16'
    }
    elseif ($Mode -eq 'Screen') {
        $trainSeeds = @('101', '202', '303')
        $evalSeeds = @('6101', '6202', '6303')
        $trainSteps = '20000'
        $poolSize = '3000'
        $evalInterval = '5000'
        $evalTasks = '10'
        $nominalTasks = '300'
        $stressTasks = '50'
        $nominalParticles = '200'
        $stressParticles = '80'
        $bootstrap = '500'
        $scenarioNames = @('nominal', 'analyte_low', 'analyte_high', 'actuator_random', 'noise_005', 'noise_010', 'bias_010', 'drift_001', 'partial_response', 'low_conc_noise', 'partial_bias')
        $workers = [Math]::Min(3, $workers)
        $ppoBatch = '1024'
        $ppoEpochs = '3'
        $sacWarmup = '500'
        $sacBatch = '128'
    }
    else {
        $trainSeeds = @('101', '202', '303', '404', '555')
        $evalSeeds = @('7101', '7202', '7303', '7404', '7555')
        $trainSteps = '60000'
        $poolSize = '8000'
        $evalInterval = '10000'
        $evalTasks = '20'
        $nominalTasks = '1000'
        $stressTasks = '200'
        $nominalParticles = '500'
        $stressParticles = '100'
        $bootstrap = '2000'
        $scenarioNames = @('nominal', 'analyte_low', 'analyte_high', 'volume_small', 'volume_large', 'titrant_low', 'actuator_under', 'actuator_over', 'actuator_random', 'noise_005', 'noise_010', 'bias_010', 'drift_001', 'partial_response', 'tetraprotic', 'out_of_range', 'close_pka', 'low_conc_noise', 'high_conc_under', 'large_volume_drift', 'partial_bias', 'tetra_noise', 'close_random_actuator')
        $ppoBatch = '2048'
        $ppoEpochs = '4'
        $sacWarmup = '1000'
        $sacBatch = '256'
    }

    $trainingDir = Join-Path $script:RunDir 'training'
    $evaluationDir = Join-Path $script:RunDir 'evaluation'
    $resumeArg = if ($Force) { @() } else { @('--resume') }
    Invoke-Step 'train' 'train_candidates.py' (@(
        '--imitation-weights', (Join-Path $Root 'models\imitation.pth'),
        '--seeds'
    ) + $trainSeeds + @(
        '--train-steps', $trainSteps,
        '--training-pool-size', $poolSize,
        '--eval-interval', $evalInterval,
        '--eval-tasks', $evalTasks,
        '--ppo-batch-steps', $ppoBatch,
        '--ppo-epochs', $ppoEpochs,
        '--sac-warmup', $sacWarmup,
        '--sac-batch-size', $sacBatch,
        '--device', $trainDevice,
        '--output-dir', $trainingDir
    ) + $resumeArg)

    Invoke-Step 'evaluate' 'evaluate_candidates.py' (@(
        '--candidate-dir', (Join-Path $trainingDir 'models'),
        '--train-seeds'
    ) + $trainSeeds + @(
        '--eval-seeds'
    ) + $evalSeeds + @(
        '--nominal-tasks', $nominalTasks,
        '--stress-tasks', $stressTasks,
        '--nominal-particles', $nominalParticles,
        '--stress-particles', $stressParticles,
        '--bootstrap-iterations', $bootstrap,
        '--scenarios'
    ) + $scenarioNames + @(
        '--device', 'cpu',
        '--workers', [string]$workers,
        '--output-dir', $evaluationDir
    ))

    Invoke-Step 'report' 'build_report.py' @(
        '--training-dir', $trainingDir,
        '--evaluation-dir', $evaluationDir,
        '--output-dir', $script:RunDir
    )
    Set-Content -LiteralPath (Join-Path $script:RunDir ("RUN_COMPLETE_{0}.txt" -f $Mode.ToUpperInvariant())) -Value (Get-Date -Format 'o') -Encoding ASCII
    Remove-Item -LiteralPath (Join-Path $script:RunDir 'RUN_FAILED.txt') -Force -ErrorAction SilentlyContinue
    Write-RunLog "RL-versus-Bayesian challenge completed: $Mode"
}
catch {
    if ($script:RunDir) {
        Write-RunLog ("FAILED: " + $_.Exception.Message)
        Set-Content -LiteralPath (Join-Path $script:RunDir 'RUN_FAILED.txt') -Value $_.Exception.ToString() -Encoding UTF8
    }
    else {
        Write-Host $_.Exception.ToString()
    }
    exit 1
}
finally {
    if ($script:LockHandle) {
        $script:LockHandle.Dispose()
        $script:LockHandle = $null
    }
    Remove-Item -LiteralPath $script:LockFile -Force -ErrorAction SilentlyContinue
}
