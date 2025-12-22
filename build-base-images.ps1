<#
.SYNOPSIS
    Build and push Jeeves base images to Docker Hub

.DESCRIPTION
    This script builds the base images (system deps + pip packages) and optionally
    pushes them to Docker Hub. Run this when requirements.txt changes.

.PARAMETER Registry
    Docker Hub username or registry (default: "jeeves")

.PARAMETER Push
    Push images to Docker Hub after building

.PARAMETER Services
    Which services to build (default: all)

.EXAMPLE
    .\build-base-images.ps1

.EXAMPLE
    .\build-base-images.ps1 -Registry "yourusername" -Push

.EXAMPLE
    .\build-base-images.ps1 -Services "memory"
#>

param(
  [string]$Registry = "ianwesterfield",
  [switch]$Push = $true,
  [string[]]$Services = @("memory", "extractor", "pragmatics")
)

$ErrorActionPreference = "Stop"

$ServiceConfig = @{
  "memory"     = @{
    Context    = "tools-api"
    Dockerfile = "memory/Dockerfile.base"
    ImageName  = "jeeves-memory-base"
  }
  "extractor"  = @{
    Context    = "tools-api/extractor"
    Dockerfile = "Dockerfile.base"
    ImageName  = "jeeves-extractor-base"
  }
  "pragmatics" = @{
    Context    = "tools-api/pragmatics"
    Dockerfile = "Dockerfile.base"
    ImageName  = "jeeves-pragmatics-base"
  }
}

Write-Host ""
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "  Jeeves Base Image Builder"  -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "Registry: $Registry"
Write-Host "Services: $($Services -join ', ')"
Write-Host "Push: $Push"
Write-Host ""

$startTime = Get-Date

foreach ($service in $Services) {
  if (-not $ServiceConfig.ContainsKey($service)) {
    Write-Host "Unknown service: $service" -ForegroundColor Red
    continue
  }

  $config = $ServiceConfig[$service]
  $imageName = "$Registry/$($config.ImageName):latest"

  Write-Host ""
  Write-Host "----------------------------------------" -ForegroundColor Yellow
  Write-Host "Building: $imageName" -ForegroundColor Yellow
  Write-Host "----------------------------------------" -ForegroundColor Yellow

  $buildStart = Get-Date

  Push-Location $config.Context
  try {
    docker build -f $config.Dockerfile -t $imageName .
    if ($LASTEXITCODE -ne 0) {
      throw "Build failed for $service"
    }
  }
  finally {
    Pop-Location
  }

  $buildTime = (Get-Date) - $buildStart
  Write-Host "Built $service in $([math]::Round($buildTime.TotalSeconds, 1))s" -ForegroundColor Green

  if ($Push) {
    Write-Host "Pushing: $imageName" -ForegroundColor Yellow
    docker push $imageName
    if ($LASTEXITCODE -ne 0) {
      throw "Push failed for $service"
    }
    Write-Host "Pushed $service" -ForegroundColor Green
  }
}

$totalTime = (Get-Date) - $startTime

Write-Host ""
Write-Host "========================================"  -ForegroundColor Green
Write-Host "  Build Complete!"  -ForegroundColor Green
Write-Host "========================================"  -ForegroundColor Green
Write-Host "Total time: $([math]::Round($totalTime.TotalMinutes, 1)) minutes"
Write-Host ""

if (-not $Push) {
  Write-Host "To push images to Docker Hub:" -ForegroundColor Yellow
  Write-Host '  .\build-base-images.ps1 -Registry "yourusername" -Push' -ForegroundColor Yellow
  Write-Host ""
}

Write-Host "To use these images in docker-compose:" -ForegroundColor Cyan
Write-Host "  docker compose up -d --build" -ForegroundColor Cyan
Write-Host ""
