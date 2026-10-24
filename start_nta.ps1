# Запуск Notice-to-Action с локальной конфигурацией
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Локальное окружение не найдено: .venv\Scripts\python.exe'
}

if (-not (Test-Path Env:NTA_PROVIDER_API_KEY)) {
    $envFile = Join-Path $PSScriptRoot '.env'
    if (Test-Path -LiteralPath $envFile -PathType Leaf) {
        Write-Host 'Локальная конфигурация .env будет загружена сервером.'
    } else {
        $secureKey = Read-Host -Prompt 'Gemini API key (Enter для демо без AI)' -AsSecureString
        if ($secureKey.Length -gt 0) {
            $credential = New-Object System.Management.Automation.PSCredential('Gemini', $secureKey)
            $env:NTA_PROVIDER_API_KEY = $credential.GetNetworkCredential().Password
            $keySetByScript = $true
        }
    }
} else {
    Write-Host 'Используется ключ из переменной окружения процесса.'
}

Write-Host ''
Write-Host 'Запускаю сервер: http://127.0.0.1:8000/'
Write-Host 'Для остановки нажми Ctrl+C.'
Write-Host ''

try {
    & $python -m uvicorn app:app --host 127.0.0.1 --port 8000
}
finally {
    if ($keySetByScript) {
        Remove-Item Env:NTA_PROVIDER_API_KEY -ErrorAction SilentlyContinue
    }
    Remove-Variable secureKey, credential -ErrorAction SilentlyContinue
}
