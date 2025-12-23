# Train the 4-class intent classifier
# Run from the pragmatics/static directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Jeeves Intent Classifier Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "train_intent_classifier.py")) {
    Write-Host "ERROR: Run this from layers/pragmatics/static/" -ForegroundColor Red
    Write-Host "  cd layers/pragmatics/static" -ForegroundColor Yellow
    Write-Host "  .\train_intent.ps1" -ForegroundColor Yellow
    exit 1
}

# Check Python
Write-Host "Checking Python environment..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    exit 1
}

# Check for required packages
Write-Host "Checking dependencies..." -ForegroundColor Yellow
python -c "import torch; import transformers; import sklearn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install torch transformers scikit-learn numpy
}

# Check GPU
Write-Host ""
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}') if torch.cuda.is_available() else None"

# Run training
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

python train_intent_classifier.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Training Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Model saved to: ./distilbert_intent" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Rebuild pragmatics container:" -ForegroundColor White
    Write-Host "     docker compose up -d --build pragmatics_api" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. Test the new classifier:" -ForegroundColor White
    Write-Host "     curl -X POST http://localhost:8001/api/pragmatics/classify -H 'Content-Type: application/json' -d '{\"text\": \"list files in workspace\"}'" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "Training failed!" -ForegroundColor Red
    exit 1
}
