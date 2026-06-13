# Run baselines
Write-Host "Running Chicago baselines..."
python scripts/baselines.py data=chicago
Write-Host "Running NYC baselines..."
python scripts/baselines.py data=nyc

# Train models
Write-Host "Training Chicago model..."
python scripts/train.py data=chicago
Write-Host "Training NYC model..."
python scripts/train.py data=nyc

# Evaluate trained models
Write-Host "Evaluating Chicago model..."
python scripts/evaluate_trained.py --checkpoint auto --data chicago
Write-Host "Evaluating NYC model..."
python scripts/evaluate_trained.py --checkpoint auto --data nyc

Write-Host "ALL RUNS COMPLETED SUCCESSFULLY"

# Evaluate fairness audit
Write-Host 'Running Chicago fairness audit...'
python scripts/evaluate_fairness.py --data chicago --checkpoint auto
Write-Host 'Running NYC fairness audit...'
python scripts/evaluate_fairness.py --data nyc --checkpoint auto
