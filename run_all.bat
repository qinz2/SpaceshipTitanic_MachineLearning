@echo off
REM Spaceship Titanic Project - Run All Scripts
REM Usage: Double-click this file or run "run_all.bat" in command line
REM Note: Please ensure all dependencies are installed (pip install -r requirements.txt)

echo Spaceship Titanic Project - Starting Execution
echo.

echo [Step 1/7] Data exploration and preprocessing...
python code/01_data_exploration_and_preprocessing.py
if %errorlevel% neq 0 (
    echo Error: Step 1 execution failed
    pause
    exit /b %errorlevel%
)
echo Step 1 completed!
echo.

echo [Step 2/7] Missing value analysis and processing...
python code/02_missing_value_analysis_and_processing.py
if %errorlevel% neq 0 (
    echo Error: Step 2 execution failed
    pause
    exit /b %errorlevel%
)
echo Step 2 completed!
echo.

echo [Step 3/7] Feature engineering and selection...
python code/03_feature_engineering_and_selection.py
if %errorlevel% neq 0 (
    echo Error: Step 3 execution failed
    pause
    exit /b %errorlevel%
)
echo Step 3 completed!
echo.

echo [Step 4/7] Data preprocessing and model selection...
python code/04_data_preprocessing_and_model_selection.py
if %errorlevel% neq 0 (
    echo Error: Step 4 execution failed
    pause
    exit /b %errorlevel%
)
echo Step 4 completed!
echo.

echo [Step 5/7] Cross-validation and hyperparameter tuning (this step may take 10-30 minutes)...
python code/05_cross_validation_and_hyperparameter_tuning.py
if %errorlevel% neq 0 (
    echo Error: Step 5 execution failed
    pause
    exit /b %errorlevel%
)
echo Step 5 completed!
echo.

echo [Step 6/7] Model evaluation and feature analysis...
python code/06_model_evaluation_and_feature_analysis.py
if %errorlevel% neq 0 (
    echo Error: Step 6 execution failed
    pause
    exit /b %errorlevel%
)
echo Step 6 completed!
echo.

echo [Step 7/7] Generate test set predictions...
python code/07_generate_test_predictions.py
if %errorlevel% neq 0 (
    echo Error: Step 7 execution failed
    pause
    exit /b %errorlevel%
)
echo Step 7 completed!
echo.

echo All steps completed successfully!
echo.
echo Generated files:
echo - Submission file: result/submission.csv
echo - Detailed predictions: result/detailed_predictions.csv
echo - Visualization charts: result/*.png
echo - Evaluation results: result/*.csv
echo.
echo Next step: Submit result/submission.csv to Kaggle
echo.
pause
