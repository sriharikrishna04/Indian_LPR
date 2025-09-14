#!/usr/bin/env python3
"""
Script to prepare dataset for LPRNet training
Extracts license plates from images using detection results
"""

import os
import cv2
import argparse
from pathlib import Path
import shutil
import random

def extract_license_plates(images_dir, labels_dir, output_dir, target_size=(94, 24)):
    """
    Extract license plates from images using YOLO labels
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels
        output_dir: Output directory for cropped license plates
        target_size: Target size for LPRNet (width, height)
    """
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in images_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images")
    
    extracted_count = 0
    skipped_count = 0
    
    for img_file in image_files:
        # Find corresponding label file
        label_file = labels_path / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            print(f"Warning: No label file found for {img_file.name}")
            skipped_count += 1
            continue
        
        # Read image
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Could not read image {img_file.name}")
                skipped_count += 1
                continue
            
            img_height, img_width = img.shape[:2]
        except Exception as e:
            print(f"Warning: Error reading image {img_file.name}: {e}")
            skipped_count += 1
            continue
        
        # Read label file
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: Error reading label file {label_file.name}: {e}")
            skipped_count += 1
            continue
        
        # Extract license plates
        plate_count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                yolo_class, center_x, center_y, width, height = map(float, parts)
                
                # Convert normalized coordinates to absolute coordinates
                abs_center_x = center_x * img_width
                abs_center_y = center_y * img_height
                abs_width = width * img_width
                abs_height = height * img_height
                
                # Convert center format to corner format
                x1 = int(abs_center_x - abs_width / 2)
                y1 = int(abs_center_y - abs_height / 2)
                x2 = int(abs_center_x + abs_width / 2)
                y2 = int(abs_center_y + abs_height / 2)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop license plate
                plate_img = img[y1:y2, x1:x2]
                
                if plate_img.size == 0:
                    continue
                
                # Resize to target size
                plate_img = cv2.resize(plate_img, target_size)
                
                # Save cropped plate
                output_filename = f"{img_file.stem}_plate_{plate_count}.jpg"
                output_filepath = output_path / output_filename
                cv2.imwrite(str(output_filepath), plate_img)
                
                plate_count += 1
                extracted_count += 1
                
            except Exception as e:
                print(f"Warning: Error processing line in {label_file.name}: {line}, Error: {e}")
                continue
        
        if plate_count == 0:
            print(f"Warning: No license plates found in {img_file.name}")
            skipped_count += 1
    
    print(f"\nLicense plate extraction completed!")
    print(f"Extracted: {extracted_count} license plates")
    print(f"Skipped: {skipped_count} images")
    print(f"Output directory: {output_path}")

def create_lprnet_structure(output_dir, train_ratio=0.85):
    """
    Create train/test structure for LPRNet
    """
    output_path = Path(output_dir)
    
    # Get all extracted plates
    plate_files = list(output_path.glob("*.jpg"))
    random.shuffle(plate_files)
    
    # Split into train/test
    split_idx = int(len(plate_files) * train_ratio)
    train_files = plate_files[:split_idx]
    test_files = plate_files[split_idx:]
    
    # Create directories
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Move files
    for f in train_files:
        shutil.move(str(f), str(train_dir / f.name))
    
    for f in test_files:
        shutil.move(str(f), str(test_dir / f.name))
    
    print(f"LPRNet dataset structure created:")
    print(f"Train: {len(train_files)} plates -> {train_dir}")
    print(f"Test: {len(test_files)} plates -> {test_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for LPRNet training')
    parser.add_argument('--images_dir', required=True, help='Directory containing images')
    parser.add_argument('--labels_dir', required=True, help='Directory containing YOLO format labels')
    parser.add_argument('--output_dir', required=True, help='Output directory for cropped plates')
    parser.add_argument('--target_size', nargs=2, type=int, default=[94, 24], 
                       help='Target size for LPRNet (width height)')
    parser.add_argument('--create_structure', action='store_true', 
                       help='Create train/test structure')
    parser.add_argument('--train_ratio', type=float, default=0.85, 
                       help='Ratio for training set')
    
    args = parser.parse_args()
    
    # Extract license plates
    extract_license_plates(
        args.images_dir,
        args.labels_dir,
        args.output_dir,
        tuple(args.target_size)
    )
    
    # Create structure if requested
    if args.create_structure:
        create_lprnet_structure(args.output_dir, args.train_ratio)

if __name__ == "__main__":
    main()
