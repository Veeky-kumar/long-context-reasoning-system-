"""
KDSH 2026 Track A - Automated Evaluation Pipeline
Runs complete workflow: train → analyze → test → validate
"""

import os
import sys
import subprocess

def run_command(command, allow_failure=False):
    """
    Runs a shell command and prints output clearly.
    Exits on failure unless allow_failure=True.
    """
    print(f"\n" + "="*60)
    print(f"🚀 RUNNING: {command}")
    print("="*60)
    
    # Run command and wait for completion
    result = subprocess.run(command, shell=True)
    
    # Check if command failed
    if result.returncode != 0:
        if allow_failure:
            print(f"\n⚠️  Command returned exit code {result.returncode} (continuing anyway)")
            return False
        else:
            print(f"\n❌ ERROR: Command failed with exit code {result.returncode}")
            print("Stopping pipeline.")
            sys.exit(1)
    else:
        print("\n✓ Step Success")
        return True

def ensure_directories():
    """Create necessary output directories if they don't exist."""
    os.makedirs('/app/output', exist_ok=True)
    print("✓ Output directory ready")

def check_prerequisites():
    """Verify required files exist before starting pipeline."""
    required_files = [
        '/app/test.csv',
        '/app/solution.py',
        '/app/helpers.py'
    ]
    
    # train.csv is optional
    has_train = os.path.exists('/app/train.csv')
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("❌ Missing required files:")
        for f in missing:
            print(f"   - {f}")
        sys.exit(1)
    
    # Check novels directory
    if not os.path.exists('/app/novels') or not os.listdir('/app/novels'):
        print("❌ ERROR: novels/ directory is empty or missing")
        print("   Please place novel .txt files in the novels/ directory")
        sys.exit(1)
    
    print("✓ Prerequisites checked")
    if has_train:
        print("✓ Training data available")
    else:
        print("⚠️  No training data found (will skip training analysis)")
    
    return has_train

def main():
    """
    Main pipeline orchestrator.
    Executes full workflow with error handling.
    """
    print("\n" + "#"*60)
    print("      KDSH 2026 INTERNAL PIPELINE AUTOMATION")
    print("#"*60)
    
    # Setup
    ensure_directories()
    has_train = check_prerequisites()
    
    # --- STEP 1: Training Data Evaluation (Optional) ---
    if has_train:
        print("\n[Step 1/4] Generating predictions for Training Data...")
        success = run_command(
            "python solution.py "
            "--test /app/train.csv "
            "--novels /app/novels "
            "--output /app/output/train_predictions.csv",
            allow_failure=True
        )
        
        if success:
            # --- STEP 2: Accuracy Analysis (Optional) ---
            print("\n[Step 2/4] Analyzing Accuracy Score...")
            run_command(
                "python helpers.py analyze "
                "/app/output/train_predictions.csv "
                "/app/train.csv",
                allow_failure=True  # Don't fail if labels aren't available
            )
        else:
            print("\n⚠️  Skipping accuracy analysis (training prediction failed)")
    else:
        print("\n[Steps 1-2 SKIPPED] No training data available")
    
    # --- STEP 3: Test Data Inference ---
    step_num = 3 if has_train else 1
    print(f"\n[Step {step_num}/{'4' if has_train else '2'}] Running Inference on Test Data...")
    run_command(
        "python solution.py "
        "--test /app/test.csv "
        "--novels /app/novels "
        "--output /app/output/submission.csv"
    )
    
    # --- STEP 4: Submission Validation ---
    step_num = 4 if has_train else 2
    print(f"\n[Step {step_num}/{'4' if has_train else '2'}] Validating Submission File Format...")
    run_command(
        "python helpers.py validate "
        "/app/output/submission.csv "
        "/app/test.csv"
    )
    
    # Success message
    print("\n" + "#"*60)
    print("✅ PIPELINE COMPLETE!")
    print("#"*60)
    print("\nGenerated files:")
    if has_train and os.path.exists('/app/output/train_predictions.csv'):
        print("  📄 /app/output/train_predictions.csv")
    print("  📄 /app/output/submission.csv")
    print("\nTo access files, check the 'output' folder")
    print("mapped to your host machine.")
    print("#"*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)