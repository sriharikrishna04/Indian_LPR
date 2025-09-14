#!/usr/bin/env python3
"""
Quick fix for Kaggle environment - run dataset conversion directly
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🔧 KAGGLE QUICK FIX - Dataset Conversion")
    print("="*50)
    
    # Set paths
    dataset_path = "/kaggle/working/Indian_LPR/lp_dataset"
    output_path = "/kaggle/working/Indian_LPR/kaggle_implementation"
    script_dir = "/kaggle/working/Indian_LPR"
    
    # Change to the correct directory
    os.chdir(script_dir)
    print(f"📁 Changed to directory: {os.getcwd()}")
    
    # Run conversion
    convert_cmd = f"""python convert_dataset.py \
        --images_dir "{dataset_path}/images" \
        --labels_dir "{dataset_path}/labels" \
        --output_file "{output_path}/processed/fcos_dataset.txt" \
        --class_id 0 \
        --split \
        --train_ratio 0.85"""
    
    print(f"🔄 Running conversion command...")
    print(f"Command: {convert_cmd}")
    
    try:
        result = subprocess.run(convert_cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Dataset conversion completed!")
        print("Output:", result.stdout)
        
        # Check if files were created
        train_file = f"{output_path}/processed/fcos_dataset_train.txt"
        val_file = f"{output_path}/processed/fcos_dataset_val.txt"
        
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                train_lines = len(f.readlines())
            print(f"📊 Training samples: {train_lines}")
        
        if os.path.exists(val_file):
            with open(val_file, 'r') as f:
                val_lines = len(f.readlines())
            print(f"📊 Validation samples: {val_lines}")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e.stderr}")
        return False

if __name__ == "__main__":
    main()
