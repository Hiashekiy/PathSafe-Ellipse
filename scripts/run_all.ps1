# End-to-end pipeline driver (S0 -> training/test).
# Threads are pinned to 1 per shard: the per-shard work is many small NumPy
# operations, so BLAS/OpenMP thread pools only add contention.
param(
    [int]$Shards = 10,
    [int]$PathsPerMap = 30,
    [int]$Epochs = 40,
    [switch]$SkipGeneration
)
$ErrorActionPreference = "Stop"
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"
Set-Location "D:\ProjectDirectory\PathSafe-Ellipse"

Write-Host "== S0/S1: environment + map inventory + inflation =="
& $py scripts\inspect_maps.py

Write-Host "== geometry self-checks =="
& $py scripts\test_geometry.py
if ($LASTEXITCODE -ne 0) { throw "geometry self-checks failed" }

if (-not $SkipGeneration) {
    Write-Host "== S2-S4: sharded path + sample generation ($Shards shards) =="
    $env:OMP_NUM_THREADS = 1; $env:MKL_NUM_THREADS = 1
    $env:OPENBLAS_NUM_THREADS = 1; $env:NUMEXPR_NUM_THREADS = 1
    $jobs = @()
    for ($i = 0; $i -lt $Shards; $i++) {
        $jobs += Start-Job -ScriptBlock {
            param($py, $i, $n, $ppm)
            $env:OMP_NUM_THREADS = 1; $env:MKL_NUM_THREADS = 1
            $env:OPENBLAS_NUM_THREADS = 1; $env:NUMEXPR_NUM_THREADS = 1
            Set-Location "D:\ProjectDirectory\PathSafe-Ellipse"
            & $py -u scripts\generate_dataset.py --stage samples --paths-per-map $ppm `
                --shard $i --num-shards $n --workers 1 *> "logs\shards\run_shard_$i.log" 2>&1
        } -ArgumentList $py, $i, $Shards, $PathsPerMap
    }
    $jobs | Wait-Job | Out-Null
    $jobs | Remove-Job
}

Write-Host "== S5: filter + map-group split + NPZ =="
& $py scripts\generate_dataset.py --stage assemble

Write-Host "== independent validation =="
& $py scripts\validate_dataset.py
& $py scripts\make_report.py
& $py scripts\visualize_samples.py

Write-Host "== training =="
& $py scripts\train_ellipse_net.py --epochs $Epochs --batch-size 256

Write-Host "== test evaluation =="
& $py scripts\evaluate_ellipse_net.py --split test

Write-Host "done. see reports\ and outputs\"
