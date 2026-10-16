param(
    [string]$ProjectDir = (Get-Location).Path,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectDir "artifacts\diversity-smoke"
}

Set-Location -LiteralPath $ProjectDir
python -m pip install -e ".[test]"
python -m pytest -q
txnagentbench plan --config configs\diversity_smoke.json
txnagentbench run --config configs\diversity_smoke.json --output $OutputDir --allow-scripted
txnagentbench validate --output $OutputDir
txnagentbench analyze --output $OutputDir
Write-Host "Engineering gates passed: $OutputDir"
