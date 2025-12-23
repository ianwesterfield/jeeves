# sync-filter.ps1
# Syncs the Jeeves filter from disk to Open WebUI

param(
  [string]$OpenWebUIUrl = "http://localhost:8180",
  [string]$FilterFile = "$PSScriptRoot\..\filters\jeeves.filter.py",
  [string]$FilterId = "api",  # The function ID in Open WebUI
  [string]$ApiKey = $env:OPENWEBUI_API_KEY
)

$ErrorActionPreference = "Stop"

# Check if API key is set
if (-not $ApiKey) {
  Write-Host "ERROR: OPENWEBUI_API_KEY environment variable not set" -ForegroundColor Red
  Write-Host ""
  Write-Host "To get your API key:"
  Write-Host "  1. Open WebUI -> Settings -> Account"
  Write-Host "  2. Copy your API key"
  Write-Host "  3. Set environment variable:"
  Write-Host '     $env:OPENWEBUI_API_KEY = "your-key-here"'
  Write-Host ""
  exit 1
}

# Read filter content
$filterPath = Resolve-Path $FilterFile
Write-Host "Reading filter from: $filterPath" -ForegroundColor Cyan
$filterContent = Get-Content -Path $filterPath -Raw

# First, get the existing function to preserve metadata
Write-Host "Fetching existing function '$FilterId'..." -ForegroundColor Cyan
$headers = @{
  "Authorization" = "Bearer $ApiKey"
  "Content-Type"  = "application/json"
}

try {
  $existingFunc = Invoke-RestMethod -Uri "$OpenWebUIUrl/api/v1/functions/id/$FilterId" -Method Get -Headers $headers
  Write-Host "Found function: $($existingFunc.name)" -ForegroundColor Green
}
catch {
  Write-Host "ERROR: Could not fetch function '$FilterId'" -ForegroundColor Red
  Write-Host "Response: $_" -ForegroundColor Yellow
  Write-Host ""
  Write-Host "Make sure the filter exists in Open WebUI with ID '$FilterId'"
  exit 1
}

# Update the function content
Write-Host "Updating function content..." -ForegroundColor Cyan

$updatePayload = @{
  id      = $FilterId
  name    = $existingFunc.name
  content = $filterContent
  meta    = $existingFunc.meta
} | ConvertTo-Json -Depth 10

try {
  $result = Invoke-RestMethod -Uri "$OpenWebUIUrl/api/v1/functions/id/$FilterId/update" -Method Post -Headers $headers -Body $updatePayload
  Write-Host ""
  Write-Host "SUCCESS: Filter updated!" -ForegroundColor Green
  Write-Host "  Name: $($result.name)"
  Write-Host "  ID: $($result.id)"
  Write-Host "  Updated: $($result.updated_at)"
}
catch {
  Write-Host "ERROR: Failed to update function" -ForegroundColor Red
  Write-Host "Response: $_" -ForegroundColor Yellow
  exit 1
}
