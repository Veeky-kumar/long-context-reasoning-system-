@echo off
echo ========================================================
echo 🚀 KDSH 2026 AUTO-RUNNER (Windows)
echo ========================================================

echo.
echo [1/5] Cleaning and Rebuilding Docker Environment...
docker-compose down
docker-compose build
if %errorlevel% neq 0 (
    echo ❌ Build failed! Stopping.
    pause
    exit /b %errorlevel%
)
docker-compose up -d

echo.
echo [2/5] Running Training Set (Calculating Accuracy)...
docker-compose run evaluator python solution.py --test /app/train.csv --output /app/output/train_predictions.csv

echo.
echo [3/5] Analyzing Training Results...
docker-compose run evaluator python helpers.py analyze /app/output/train_predictions.csv /app/train.csv

echo.
echo [4/5] Running Test Set (Generating Submission)...
docker-compose run evaluator python solution.py --test /app/test.csv --output /app/output/submission.csv

echo.
echo [5/5] Validating Submission Format...
docker-compose run evaluator python helpers.py validate /app/output/submission.csv /app/test.csv

echo.
echo ========================================================
echo ✅ ALL TASKS COMPLETED SUCCESSFULLY!
echo ========================================================
pause