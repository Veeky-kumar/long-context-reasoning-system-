"""
KDSH 2026 Track A - Validation and Analysis Helper Script
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

def validate_submission(submission_path: str, test_path: str):
    print("\n" + "="*60)
    print("VALIDATING SUBMISSION FORMAT")
    print("="*60)
    
    try:
        submission = pd.read_csv(submission_path)
        test = pd.read_csv(test_path)
        
        # Normalize column names
        submission.columns = [c.strip() for c in submission.columns]
        
        # Check required columns
        if 'Story ID' not in submission.columns or 'Prediction' not in submission.columns:
            print(f"❌ ERROR: Missing columns. Found: {list(submission.columns)}")
            print("   Required: ['Story ID', 'Prediction']")
            return False
            
        # Check IDs
        sub_ids = set(submission['Story ID'].astype(str))
        test_ids = set(test['id'].astype(str))
        
        if not sub_ids == test_ids:
            print(f"❌ ERROR: ID mismatch.")
            print(f"   Missing in submission: {list(test_ids - sub_ids)[:3]}")
            return False
            
        # Check Predictions
        if not submission['Prediction'].isin([0, 1]).all():
            print("❌ ERROR: Predictions must be 0 or 1")
            return False
            
        print("✓ Format is valid")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def analyze_accuracy(predictions_path: str, ground_truth_path: str):
    print("\n" + "="*60)
    print("ANALYZING ACCURACY")
    print("="*60)
    
    try:
        pred_df = pd.read_csv(predictions_path)
        gt_df = pd.read_csv(ground_truth_path)
        
        # 1. Clean IDs
        pred_df['id_clean'] = pred_df['Story ID'].astype(str).str.strip()
        gt_df['id_clean'] = gt_df['id'].astype(str).str.strip()
        
        # 2. Clean Labels (The Fix for 0% Accuracy)
        # Handle 'consistent'/'contradict' text labels case-insensitively
        label_map = {
            'consistent': 1, 
            'contradict': 0,
            '1': 1,
            '0': 0,
            1: 1,
            0: 0
        }
        
        # Ensure label column exists and clean it
        label_col = next((c for c in gt_df.columns if c.lower() in ['label', 'consistency']), None)
        if not label_col:
            print("⚠️ No label column found in training data.")
            return False

        gt_df['label_clean'] = gt_df[label_col].astype(str).str.strip().str.lower().map(label_map)
        
        # Drop rows where label couldn't be parsed
        valid_gt = gt_df.dropna(subset=['label_clean'])
        
        if len(valid_gt) == 0:
            print("❌ ERROR: Could not parse any labels from training data.")
            print(f"   Sample raw labels: {gt_df[label_col].unique()[:5]}")
            return False

        # 3. Merge
        merged = pd.merge(pred_df, valid_gt, on='id_clean', how='inner')
        
        if len(merged) == 0:
            print("❌ ERROR: No IDs matched between files.")
            return False

        # 4. Calculate Stats
        y_true = merged['label_clean'].astype(int)
        y_pred = merged['Prediction'].astype(int)
        
        accuracy = (y_true == y_pred).mean()
        
        # Confusion Matrix
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        print(f"\n✅ ACCURACY: {accuracy:.2%} ({len(merged)} samples)")
        print(f"\nConfusion Matrix:")
        print(f"Actual 0 (Contradict) : Correct: {tn:<3} | Wrong: {fp:<3}")
        print(f"Actual 1 (Consistent) : Correct: {tp:<3} | Wrong: {fn:<3}")
        
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    val = subparsers.add_parser('validate')
    val.add_argument('submission')
    val.add_argument('test')
    
    ana = subparsers.add_parser('analyze')
    ana.add_argument('predictions')
    ana.add_argument('ground_truth')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        validate_submission(args.submission, args.test)
    elif args.command == 'analyze':
        analyze_accuracy(args.predictions, args.ground_truth)

if __name__ == "__main__":
    main()