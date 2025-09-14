#!/usr/bin/env python3
"""
Quick Start Script for Kaggle Dataset
Simple setup to get you started immediately
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 KAGGLE DATASET QUICK START")
    print("="*50)
    
    # Get dataset path from user
    print("\n📁 Please provide the path to your Kaggle dataset:")
    print("   (The folder containing 'images' and 'labels' subfolders)")
    
    dataset_path = input("Dataset path: ").strip()
    
    if not dataset_path:
        print("❌ No path provided. Exiting.")
        return
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Path does not exist: {dataset_path}")
        return
    
    # Check if images and labels folders exist
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not images_dir.exists():
        print(f"❌ Images folder not found: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"❌ Labels folder not found: {labels_dir}")
        return
    
    print(f"✅ Dataset found at: {dataset_path}")
    
    # Count files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt"))
    
    print(f"📊 Found {len(image_files)} images and {len(label_files)} labels")
    
    # Create output directory
    output_dir = Path("./kaggle_training")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Convert dataset
    print("\n🔄 Step 1: Converting dataset to FCOS format...")
    convert_cmd = f'python convert_dataset.py --images_dir "{images_dir}" --labels_dir "{labels_dir}" --output_file "{output_dir}/fcos_dataset.txt" --class_id 0 --split --train_ratio 0.85'
    
    print(f"Running: {convert_cmd}")
    os.system(convert_cmd)
    
    # Step 2: Prepare LPRNet data
    print("\n🔄 Step 2: Preparing LPRNet data...")
    lpr_cmd = f'python prepare_lprnet_dataset.py --images_dir "{images_dir}" --labels_dir "{labels_dir}" --output_dir "{output_dir}/lprnet_plates" --target_size 94 24 --create_structure --train_ratio 0.85'
    
    print(f"Running: {lpr_cmd}")
    os.system(lpr_cmd)
    
    # Create training commands
    print("\n📝 Training Commands Created:")
    print("="*50)
    
    print("\n🔍 Object Detection Training (FCOS):")
    print(f'python src/object_detection/train.py --train_txt "{output_dir}/fcos_dataset_train.txt" --batch_size 8 --epochs 50 --output_path "{output_dir}/od_weights"')
    
    print("\n🔤 Character Recognition Training (LPRNet):")
    print(f'python src/License_Plate_Recognition/train_LPRNet.py --train_img_dirs "{output_dir}/lprnet_plates/train" --test_img_dirs "{output_dir}/lprnet_plates/test" --max_epoch 200 --train_batch_size 128 --save_folder "{output_dir}/lprnet_weights/"')
    
    print("\n✅ Setup completed!")
    print(f"📁 Check the results in: {output_dir}")
    
    # Ask if user wants to start training
    response = input("\n🚀 Do you want to start training now? (y/n): ")
    if response.lower() == 'y':
        print("\n🔄 Starting Object Detection Training...")
        od_cmd = f'python src/object_detection/train.py --train_txt "{output_dir}/fcos_dataset_train.txt" --batch_size 8 --epochs 50 --output_path "{output_dir}/od_weights"'
        os.system(od_cmd)
        
        print("\n🔄 Starting LPRNet Training...")
        lpr_train_cmd = f'python src/License_Plate_Recognition/train_LPRNet.py --train_img_dirs "{output_dir}/lprnet_plates/train" --test_img_dirs "{output_dir}/lprnet_plates/test" --max_epoch 200 --train_batch_size 128 --save_folder "{output_dir}/lprnet_weights/"'
        os.system(lpr_train_cmd)
        
        print("\n🎉 Training completed!")
    else:
        print("\n⏸️  Training skipped. You can run the commands above when ready.")

if __name__ == "__main__":
    main()
