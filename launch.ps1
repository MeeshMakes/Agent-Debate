$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot
& "C:\Users\FathomPC\AppData\Local\Programs\Python\Python314\python.exe" -m app.main
