# Launch the verified existing interpreter without changing the global environment.
$ErrorActionPreference = 'Stop'
$taskPython = $env:INTENT_GRAPH_PYTHON
if (-not $taskPython) {
    $taskPython = 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe'
}
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    $taskPython = (Get-Command python -ErrorAction Stop).Source
}
& $taskPython -B (Join-Path $PSScriptRoot 'run_experiment.py') @args
exit $LASTEXITCODE
