param(
    [string]$ShardsRoot = "results_shards",
    [string]$Python = "C:\Users\ZSY\Desktop\FDTD\rl_random_vs_imitation_processed_20260806\.venv\Scripts\python.exe",
    [string]$Algorithms = "ppo,a2c,reinforce",
    [int[]]$Seeds = @(101, 202, 303, 404, 555),
    [int]$TrainSteps = 25000,
    [int]$TrainingPoolSize = 5000,
    [int]$EvalTasks = 1000,
    [int]$EvalInterval = 5000,
    [int]$TorchThreads = 1,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = [System.IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python environment: $Python" }

$algorithmList = $Algorithms.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$shardsPath = Join-Path $Root $ShardsRoot
if ($Clean -and (Test-Path -LiteralPath $shardsPath)) {
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $resolvedShards = [System.IO.Path]::GetFullPath($shardsPath)
    if (-not $resolvedShards.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a shard path outside the project: $resolvedShards"
    }
    Remove-Item -LiteralPath $resolvedShards -Recurse -Force
}
New-Item -ItemType Directory -Path $shardsPath -Force | Out-Null

$launches = @()
foreach ($algorithm in $algorithmList) {
    foreach ($seed in $Seeds) {
        $name = "${algorithm}_seed${seed}"
        $output = Join-Path $shardsPath $name
        $stdout = Join-Path $shardsPath "${name}.log"
        $stderr = Join-Path $shardsPath "${name}.err"
        $arguments = @(
            "direction_assisted_rl_comparison.py",
            "--algorithms", $algorithm,
            "--seeds", $seed,
            "--train-steps", $TrainSteps,
            "--training-pool-size", $TrainingPoolSize,
            "--eval-tasks", $EvalTasks,
            "--eval-interval", $EvalInterval,
            "--torch-threads", $TorchThreads,
            "--output-dir", $output
        )
        $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $Root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
        $launches += [pscustomobject]@{
            name = $name
            pid = $process.Id
            output = [System.IO.Path]::GetFullPath($output)
            stdout = [System.IO.Path]::GetFullPath($stdout)
            stderr = [System.IO.Path]::GetFullPath($stderr)
        }
    }
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$manifest = Join-Path $shardsPath "launch_manifest_${stamp}.json"
$launches | ConvertTo-Json | Set-Content -LiteralPath $manifest -Encoding UTF8
Write-Host ("Started {0} independent algorithm/seed shards." -f $launches.Count)
$launches | Format-Table -AutoSize
