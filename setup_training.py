#!/usr/bin/env python3
"""
Complete setup script for Indian LPR training with your dataset
"""

import os
import argparse
from pathlib import Path
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print("Error:", e.stderr)
        return False

def setup_object_detection_training(images_dir, labels_dir, output_dir):
    """Setup object detection training"""
    print("\n🔍 Setting up Object Detection Training (FCOS)")
    
    # Convert dataset
    convert_cmd = f"""python convert_dataset.py \
        --images_dir "{images_dir}" \
        --labels_dir "{labels_dir}" \
        --output_file "{output_dir}/fcos_dataset.txt" \
        --class_id 0 \
        --split \
        --train_ratio 0.85"""
    
    if not run_command(convert_cmd, "Converting dataset to FCOS format"):
        return False
    
    # Create training command
    train_cmd = f"""python src/object_detection/train.py \
        --train_txt "{output_dir}/fcos_dataset_train.txt" \
        --batch_size 8 \
        --epochs 50 \
        --output_path "{output_dir}/od_weights" """
    
    print(f"\n📝 Object Detection Training Command:")
    print(train_cmd)
    
    return True

def setup_semantic_segmentation_training(images_dir, labels_dir, output_dir):
    """Setup semantic segmentation training"""
    print("\n🎯 Setting up Semantic Segmentation Training (HRNet)")
    
    # Note: This requires creating segmentation masks from bounding boxes
    print("⚠️  Note: Semantic segmentation requires segmentation masks.")
    print("You'll need to create binary masks from your bounding box labels.")
    print("Consider using the object detection approach instead for easier setup.")
    
    return True

def setup_lprnet_training(images_dir, labels_dir, output_dir):
    """Setup LPRNet training"""
    print("\n🔤 Setting up LPRNet Training")
    
    # Extract license plates
    extract_cmd = f"""python prepare_lprnet_dataset.py \
        --images_dir "{images_dir}" \
        --labels_dir "{labels_dir}" \
        --output_dir "{output_dir}/lprnet_plates" \
        --target_size 94 24 \
        --create_structure \
        --train_ratio 0.85"""
    
    if not run_command(extract_cmd, "Extracting license plates for LPRNet"):
        return False
    
    # Create training command
    train_cmd = f"""python src/License_Plate_Recognition/train_LPRNet.py \
        --train_img_dirs "{output_dir}/lprnet_plates/train" \
        --test_img_dirs "{output_dir}/lprnet_plates/test" \
        --max_epoch 200 \
        --train_batch_size 128 \
        --test_batch_size 128 \
        --save_folder "{output_dir}/lprnet_weights/" """
    
    print(f"\n📝 LPRNet Training Command:")
    print(train_cmd)
    
    return True

def create_training_scripts(output_dir):
    """Create training scripts for easy execution"""
    
    # Object Detection Training Script
    od_script = f"""#!/bin/bash
# Object Detection Training Script
echo "Starting Object Detection Training..."

python src/object_detection/train.py \\
    --train_txt "{output_dir}/fcos_dataset_train.txt" \\
    --batch_size 8 \\
    --epochs 50 \\
    --output_path "{output_dir}/od_weights"

echo "Object Detection Training Completed!"
"""
    
    with open(f"{output_dir}/train_object_detection.sh", "w") as f:
        f.write(od_script)
    
    # LPRNet Training Script
    lpr_script = f"""#!/bin/bash
# LPRNet Training Script
echo "Starting LPRNet Training..."

python src/License_Plate_Recognition/train_LPRNet.py \\
    --train_img_dirs "{output_dir}/lprnet_plates/train" \\
    --test_img_dirs "{output_dir}/lprnet_plates/test" \\
    --max_epoch 200 \\
    --train_batch_size 128 \\
    --test_batch_size 128 \\
    --save_folder "{output_dir}/lprnet_weights/"

echo "LPRNet Training Completed!"
"""
    
    with open(f"{output_dir}/train_lprnet.sh", "w") as f:
        f.write(lpr_script)
    
    print(f"\n📄 Training scripts created in {output_dir}/")
    print("  - train_object_detection.sh")
    print("  - train_lprnet.sh")

def main():
    parser = argparse.ArgumentParser(description='Setup training for Indian LPR project')
    parser.add_argument('--images_dir', required=True, help='Directory containing images')
    parser.add_argument('--labels_dir', required=True, help='Directory containing YOLO format labels')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed data and weights')
    parser.add_argument('--skip_od', action='store_true', help='Skip object detection setup')
    parser.add_argument('--skip_lpr', action='store_true', help='Skip LPRNet setup')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Setting up Indian LPR Training Pipeline")
    print(f"Images: {args.images_dir}")
    print(f"Labels: {args.labels_dir}")
    print(f"Output: {args.output_dir}")
    
    success = True
    
    # Setup Object Detection
    if not args.skip_od:
        success &= setup_object_detection_training(args.images_dir, args.labels_dir, args.output_dir)
    
    # Setup LPRNet
    if not args.skip_lpr:
        success &= setup_lprnet_training(args.images_dir, args.labels_dir, args.output_dir)
    
    # Create training scripts
    create_training_scripts(args.output_dir)
    
    if success:
        print("\n✅ Setup completed successfully!")
        print(f"\n📋 Next steps:")
        print(f"1. Review the generated training commands")
        print(f"2. Run: bash {args.output_dir}/train_object_detection.sh")
        print(f"3. Run: bash {args.output_dir}/train_lprnet.sh")
        print(f"4. Test with: python infer_objectdet.py --source your_image.jpg")
    else:
        print("\n❌ Setup completed with errors. Please check the output above.")

if __name__ == "__main__":
    main()
