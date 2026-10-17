param(
    [string]$ProjectDir = (Get-Location).Path,
    [string]$OutputDir = "",
    [string]$RecoveryOutputDir = ""
)

$ErrorActionPreference = "Stop"
if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectDir "artifacts\diversity-smoke"
}
if (-not $RecoveryOutputDir) {
    $RecoveryOutputDir = Join-Path $ProjectDir "artifacts\recovery-smoke"
}

Set-Location -LiteralPath $ProjectDir
python -m pip install -e ".[test]"
python -m pytest -q
txnagentbench plan --config configs\diversity_smoke.json
txnagentbench run --config configs\diversity_smoke.json --output $OutputDir --allow-scripted
txnagentbench validate --output $OutputDir
txnagentbench analyze --output $OutputDir
txnagentbench plan --config configs\recovery_smoke.json
txnagentbench run --config configs\recovery_smoke.json --output $RecoveryOutputDir --allow-scripted
txnagentbench validate --output $RecoveryOutputDir
txnagentbench analyze --output $RecoveryOutputDir
Write-Host "Engineering gates passed: $OutputDir and $RecoveryOutputDir"
