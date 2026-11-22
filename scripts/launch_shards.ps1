# Launch the full dataset generation as N filesystem shards (no IPC: each map
# writes its own NPZ, so the shards never need to talk to each other).
param(
    [int]$Shards = 12,
    [int]$PathsPerMap = 30
)
$ErrorActionPreference = "Stop"
$root = "D:\ProjectDirectory\PathSafe-Ellipse"
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"
New-Item -ItemType Directory -Force -Path "$root\logs\shards" | Out-Null

for ($i = 0; $i -lt $Shards; $i++) {
    $args = @(
        "scripts\generate_dataset.py",
        "--stage", "samples",
        "--paths-per-map", "$PathsPerMap",
        "--shard", "$i",
        "--num-shards", "$Shards",
        "--workers", "1"
    )
    Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $root -NoNewWindow `
        -RedirectStandardOutput "$root\logs\shards\shard_$i.log" `
        -RedirectStandardError "$root\logs\shards\shard_$i.err.log" | Out-Null
    Write-Host "launched shard $i"
}
Write-Host "all $Shards shards launched"
