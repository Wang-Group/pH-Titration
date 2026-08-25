param(
    [ValidateSet('Quick', 'Full')]
    [string]$Mode = 'Full',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root
$ModeKey = $Mode.ToLowerInvariant()
$RunDir = Join-Path $Root ("results_" + $ModeKey)
$LogFile = Join-Path $RunDir ("run_" + $ModeKey + ".log")
$ProgressFile = Join-Path $RunDir 'PROGRESS.txt'
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
if ($Force) {
    Remove-Item -LiteralPath $LogFile, $ProgressFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $RunDir 'RUN_COMPLETE.txt'), (Join-Path $RunDir 'RUN_FAILED.txt') -Force -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $RunDir -File -Filter '.done_*' -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

function Format-PortablePath([string]$Path) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $rootPrefix = $Root.TrimEnd('\') + '\'
    if ($fullPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $fullPath.Substring($rootPrefix.Length)
    }
    return $fullPath
}

function Write-RunLog([string]$Message) {
    $line = ('[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message)
    Write-Host $line
    Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
    Set-Content -LiteralPath $ProgressFile -Value $line -Encoding UTF8
}

function Test-Python([string]$Executable) {
    if (-not (Test-Path -LiteralPath $Executable)) {
        return $false
    }
    try {
        & $Executable -c "import struct, sys; raise SystemExit(0 if sys.version_info >= (3, 10) and struct.calcsize('P') * 8 == 64 else 1)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Resolve-Python {
    if ($env:PH_REVIEW_PYTHON) {
        if (Test-Python $env:PH_REVIEW_PYTHON) {
            return (Resolve-Path -LiteralPath $env:PH_REVIEW_PYTHON).Path
        }
        throw 'PH_REVIEW_PYTHON does not point to a 64-bit Python 3.10 or newer executable.'
    }
    $candidates = New-Object System.Collections.Generic.List[string]
    $localPrograms = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'Programs\Python' } else { $null }
    $userProfile = $env:USERPROFILE
    $programData = $env:ProgramData
    foreach ($known in @(
        $(if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX 'python.exe' }),
        $(if ($userProfile) { Join-Path $userProfile 'anaconda3\python.exe' }),
        $(if ($userProfile) { Join-Path $userProfile 'miniconda3\python.exe' }),
        $(if ($programData) { Join-Path $programData 'anaconda3\python.exe' }),
        $(if ($programData) { Join-Path $programData 'miniconda3\python.exe' }),
        $(if ($localPrograms) { Join-Path $localPrograms 'Python313\python.exe' }),
        $(if ($localPrograms) { Join-Path $localPrograms 'Python312\python.exe' }),
        $(if ($localPrograms) { Join-Path $localPrograms 'Python311\python.exe' }),
        $(if ($localPrograms) { Join-Path $localPrograms 'Python310\python.exe' }),
        'C:\Python313\python.exe',
        'C:\Python312\python.exe',
        'C:\Python311\python.exe',
        'C:\Python310\python.exe',
        'C:\Program Files\Python313\python.exe',
        'C:\Program Files\Python312\python.exe',
        'C:\Program Files\Python311\python.exe'
    )) {
        if ($known) { $candidates.Add($known) }
    }
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates.Add($pythonCommand.Source)
    }
    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($selector in @('-3.11', '-3.12', '-3.10', '-3.13', '-3')) {
            try {
                $resolved = (& $pyLauncher.Source $selector -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1).Trim()
                if ($resolved) { $candidates.Add($resolved) }
            }
            catch {}
        }
    }
    $validCandidates = New-Object System.Collections.Generic.List[string]
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Python $candidate) {
            $validCandidates.Add((Resolve-Path -LiteralPath $candidate).Path)
        }
    }
    foreach ($candidate in $validCandidates) {
        if (Test-Dependencies $candidate) {
            return $candidate
        }
    }
    if ($validCandidates.Count -gt 0) {
        return $validCandidates[0]
    }
    throw 'A 64-bit Python 3.10 or newer was not found. Install 64-bit Python 3.11, or set PH_REVIEW_PYTHON to python.exe.'
}

function Test-Dependencies([string]$Executable) {
    try {
        & $Executable -c "import numpy, scipy, torch" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Invoke-LiveCommand([string]$Executable, [string[]]$Arguments) {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Executable @Arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode`: $Executable $($Arguments -join ' ')"
    }
}

function Resolve-Model(
    [string]$EnvironmentVariable,
    [string]$NormalizedName,
    [string]$OriginalName,
    [string]$FallbackPattern
) {
    $explicit = [Environment]::GetEnvironmentVariable($EnvironmentVariable)
    if ($explicit -and (Test-Path -LiteralPath $explicit)) {
        return (Resolve-Path -LiteralPath $explicit).Path
    }
    foreach ($candidate in @(
        (Join-Path $Root ('models\' + $NormalizedName)),
        (Join-Path $Root ('models\' + $OriginalName)),
        (Join-Path $Root $NormalizedName),
        (Join-Path $Root $OriginalName)
    )) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    $found = Get-ChildItem -LiteralPath $Root -Recurse -File -Filter $FallbackPattern -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch '(?i)(random|smoke)' } |
        Select-Object -First 1
    if ($found) {
        return $found.FullName
    }
    return $null
}

function Invoke-Step(
    [string]$Name,
    [string]$Script,
    [string[]]$Arguments
) {
    $safeName = $Name -replace '[^A-Za-z0-9_-]', '_'
    $marker = Join-Path $RunDir ('.done_' + $safeName)
    if ((Test-Path -LiteralPath $marker) -and -not $Force) {
        Write-RunLog "SKIP $Name (completion marker exists)."
        return
    }
    Write-RunLog "START $Name"
    $started = Get-Date
    Invoke-LiveCommand $script:PythonExe (@((Join-Path $Root $Script)) + $Arguments)
    $elapsed = (Get-Date) - $started
    Set-Content -LiteralPath $marker -Value (Get-Date -Format 'o') -Encoding ASCII
    Write-RunLog ("DONE {0} in {1}" -f $Name, $elapsed)
}

function Invariant([object]$Value) {
    return [System.Convert]::ToString($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

try {
    Remove-Item -LiteralPath (Join-Path $RunDir 'RUN_COMPLETE.txt') -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $RunDir 'RUN_FAILED.txt') -Force -ErrorAction SilentlyContinue
    Write-RunLog "Starting $Mode analysis."
    $script:PythonExe = Resolve-Python
    $pythonDescription = (& $script:PythonExe -c "import platform, struct; print(platform.python_version() + ' (' + str(struct.calcsize('P') * 8) + '-bit)')" | Select-Object -Last 1).Trim()
    Write-RunLog "Using Python: $pythonDescription"

    if (-not (Test-Dependencies $script:PythonExe)) {
        $venv = Join-Path $Root '.venv'
        $venvPython = Join-Path $venv 'Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $venvPython)) {
            Write-RunLog 'Creating local Python environment.'
            Invoke-LiveCommand $script:PythonExe @('-m', 'venv', $venv)
        }
        $script:PythonExe = $venvPython
        Write-RunLog 'Installing required Python packages. This needs internet access on the first run.'
        Invoke-LiveCommand $script:PythonExe @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-LiveCommand $script:PythonExe @('-m', 'pip', 'install', '-r', (Join-Path $Root 'requirements.txt'))
    }

    $ImitationWeights = Resolve-Model 'IMITATION_WEIGHTS' 'imitation.pth' 'Volume regressor best big discrete new1 test.pth' '*imit*.pth'
    $RlWeights = Resolve-Model 'RL_WEIGHTS' 'reinforcement.pth' 'Volume regressor best big discrete new1 trained 1 test save the best.pth' '*reinfor*.pth'
    if (-not $ImitationWeights -or -not $RlWeights) {
        $missing = @(
            'The original trained model weights were not found.',
            '',
            'Place them in the models folder using either the normalized names:',
            '  imitation.pth',
            '  reinforcement.pth',
            '',
            'or the original notebook filenames listed in models\PUT_MODEL_WEIGHTS_HERE.txt.',
            '',
            'You may also set IMITATION_WEIGHTS and RL_WEIGHTS environment variables.'
        )
        $missing | Set-Content -LiteralPath (Join-Path $Root 'MISSING_MODELS.txt') -Encoding UTF8
        throw 'Model weights are missing. See MISSING_MODELS.txt.'
    }
    Remove-Item -LiteralPath (Join-Path $Root 'MISSING_MODELS.txt') -Force -ErrorAction SilentlyContinue
    Write-RunLog "Imitation weights: $(Format-PortablePath $ImitationWeights)"
    Write-RunLog "RL weights: $(Format-PortablePath $RlWeights)"

    $Device = (& $script:PythonExe -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" | Select-Object -Last 1).Trim()
    Write-RunLog "Neural-network device: $Device"
    Invoke-LiveCommand $script:PythonExe @(
        (Join-Path $Root 'validate_package.py'),
        '--imitation', $ImitationWeights,
        '--reinforcement', $RlWeights,
        '--device', $Device,
        '--output', (Join-Path $RunDir 'package_validation.json')
    )

    if ($Mode -eq 'Quick') {
        $Seeds = @(7, 8)
        $TasksPerSeed = 40
        $StressTasks = 20
        $StressScenarioSet = 'core'
        $RobustnessTasks = 10
        $Particles = 30
        $PidTrials = 1
        $PidTrainTasks = 10
        $PidEvaluationTasks = 20
        $RlTrainSteps = 20
        $RlTrainingPool = 10
        $RlEvalTasks = 10
        $RlEvalInterval = 10
        $Workers = 1
    }
    else {
        $Seeds = @(101, 202, 303, 404, 555)
        $TasksPerSeed = 3000
        $StressTasks = 1000
        $StressScenarioSet = 'full'
        $RobustnessTasks = 1000
        $Particles = 1000
        $PidTrials = 120
        $PidTrainTasks = 500
        $PidEvaluationTasks = 3000
        $RlTrainSteps = 25000
        $RlTrainingPool = 5000
        $RlEvalTasks = 1000
        $RlEvalInterval = 5000
        $Workers = if ($Device -eq 'cpu') { 8 } else { 1 }
    }
    $SeedArgs = @($Seeds | ForEach-Object { [string]$_ })
    Write-RunLog "Parallel workers: $Workers"

    $PidDir = Join-Path $RunDir 'pid_tuning'
    Invoke-Step 'pid_tuning' 'pid_tuning.py' (@(
        '--trials', [string]$PidTrials,
        '--train-tasks', [string]$PidTrainTasks,
        '--evaluation-seeds'
    ) + $SeedArgs + @(
        '--evaluation-tasks-per-seed', [string]$PidEvaluationTasks,
        '--workers', [string]$Workers,
        '--output-dir', $PidDir
    ))
    $pidSettings = Get-Content -Raw -LiteralPath (Join-Path $PidDir 'selected_pid_parameters.json') | ConvertFrom-Json

    Invoke-Step 'multiseed' 'multiseed_benchmark.py' (@(
        '--imitation-weights', $ImitationWeights,
        '--rl-weights', $RlWeights,
        '--device', $Device,
        '--seeds'
    ) + $SeedArgs + @(
        '--tasks-per-seed', [string]$TasksPerSeed,
        '--bayesian-particles', [string]$Particles,
        '--pid-kp', (Invariant $pidSettings.kp),
        '--pid-ki', (Invariant $pidSettings.ki),
        '--pid-kd', (Invariant $pidSettings.kd),
        '--pid-integral-limit', (Invariant $pidSettings.integral_limit),
        '--pid-overshoot-decay', (Invariant $pidSettings.overshoot_decay),
        '--workers', [string]$Workers,
        '--output-dir', (Join-Path $RunDir 'multiseed')
    ))

    Invoke-Step 'rl_il_stress' 'rl_il_stress_benchmark.py' (@(
        '--imitation-weights', $ImitationWeights,
        '--rl-weights', $RlWeights,
        '--device', $Device,
        '--seeds'
    ) + $SeedArgs + @(
        '--tasks-per-seed', [string]$StressTasks,
        '--scenario-set', $StressScenarioSet,
        '--output-dir', (Join-Path $RunDir 'rl_il_stress')
    ))

    Invoke-Step 'bayesian_reference' 'bayesian_robustness.py' (@(
        'reference', '--seeds'
    ) + $SeedArgs + @(
        '--tasks-per-seed', [string]$RobustnessTasks,
        '--particles', [string]$Particles,
        '--workers', [string]$Workers,
        '--output-dir', (Join-Path $RunDir 'bayesian_reference')
    ))

    Invoke-Step 'bayesian_noise' 'bayesian_robustness.py' (@(
        'noise', '--seeds'
    ) + $SeedArgs + @(
        '--tasks-per-seed', [string]$RobustnessTasks,
        '--particles', [string]$Particles,
        '--workers', [string]$Workers,
        '--output-dir', (Join-Path $RunDir 'bayesian_noise')
    ))

    Invoke-Step 'rl_algorithms' 'rl_algorithm_reward_study.py' (@(
        'algorithms',
        '--imitation-weights', $ImitationWeights,
        '--algorithms', 'reinforce', 'a2c', 'ppo',
        '--seeds'
    ) + $SeedArgs + @(
        '--train-steps', [string]$RlTrainSteps,
        '--training-pool-size', [string]$RlTrainingPool,
        '--eval-tasks', [string]$RlEvalTasks,
        '--eval-interval', [string]$RlEvalInterval,
        '--device', $Device,
        '--workers', [string]$Workers,
        '--output-dir', (Join-Path $RunDir 'rl_algorithms')
    ))

    Invoke-Step 'rl_rewards' 'rl_algorithm_reward_study.py' (@(
        'rewards',
        '--imitation-weights', $ImitationWeights,
        '--reward-algorithm', 'reinforce',
        '--seeds'
    ) + $SeedArgs + @(
        '--train-steps', [string]$RlTrainSteps,
        '--training-pool-size', [string]$RlTrainingPool,
        '--eval-tasks', [string]$RlEvalTasks,
        '--eval-interval', [string]$RlEvalInterval,
        '--device', $Device,
        '--workers', [string]$Workers,
        '--output-dir', (Join-Path $RunDir 'rl_rewards')
    ))

    Invoke-LiveCommand $script:PythonExe @(
        (Join-Path $Root 'build_result_summary.py'),
        '--run-dir', $RunDir
    )
    $completeText = @(
        "Completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
        "Mode: $Mode",
        "Summary: results_$ModeKey\RESULT_SUMMARY.md",
        "Log: results_$ModeKey\run_$ModeKey.log"
    )
    $completeText | Set-Content -LiteralPath (Join-Path $RunDir 'RUN_COMPLETE.txt') -Encoding UTF8
    Remove-Item -LiteralPath (Join-Path $RunDir 'RUN_FAILED.txt') -Force -ErrorAction SilentlyContinue
    Write-RunLog "All $Mode analyses completed successfully."
}
catch {
    $message = $_.Exception.Message
    Write-RunLog "FAILED: $message"
    Remove-Item -LiteralPath (Join-Path $RunDir 'RUN_COMPLETE.txt') -Force -ErrorAction SilentlyContinue
    @(
        "Failed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
        "Mode: $Mode",
        "Reason: $message",
        "Log: results_$ModeKey\run_$ModeKey.log"
    ) | Set-Content -LiteralPath (Join-Path $RunDir 'RUN_FAILED.txt') -Encoding UTF8
    exit 1
}
