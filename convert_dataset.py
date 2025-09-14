#!/usr/bin/env python3
"""
Dataset conversion script for Indian LPR project
Converts YOLO format dataset to FCOS format
"""

import os
import cv2
import argparse
from pathlib import Path
import random

def convert_yolo_to_fcos(images_dir, labels_dir, output_file, class_id=0):
    """
    Convert YOLO format dataset to FCOS format
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels
        output_file: Output file path for FCOS format
        class_id: Class ID for license plates (default: 0)
    """
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    if not images_path.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_path.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in images_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images")
    
    fcos_lines = []
    processed_count = 0
    skipped_count = 0
    
    for img_file in image_files:
        # Find corresponding label file
        label_file = labels_path / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            print(f"Warning: No label file found for {img_file.name}")
            skipped_count += 1
            continue
        
        # Read image to get dimensions
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
        
        # Convert YOLO format to FCOS format
        fcos_boxes = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: Invalid label format in {label_file.name}: {line}")
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
                    print(f"Warning: Invalid box in {label_file.name}: ({x1},{y1},{x2},{y2})")
                    continue
                
                fcos_boxes.append(f"{x1},{y1},{x2},{y2},{class_id}")
                
            except Exception as e:
                print(f"Warning: Error processing line in {label_file.name}: {line}, Error: {e}")
                continue
        
        if fcos_boxes:
            fcos_line = f"{img_file.absolute()} {' '.join(fcos_boxes)}"
            fcos_lines.append(fcos_line)
            processed_count += 1
        else:
            print(f"Warning: No valid boxes found in {label_file.name}")
            skipped_count += 1
    
    # Shuffle the dataset
    random.shuffle(fcos_lines)
    
    # Write to output file
    try:
        with open(output_file, 'w') as f:
            for line in fcos_lines:
                f.write(line + '\n')
        
        print(f"\nConversion completed!")
        print(f"Processed: {processed_count} images")
        print(f"Skipped: {skipped_count} images")
        print(f"Output file: {output_file}")
        
    except Exception as e:
        print(f"Error writing output file: {e}")

def split_dataset(input_file, train_ratio=0.85):
    """
    Split dataset into train and validation sets
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    # Write train file
    train_file = input_file.replace('.txt', '_train.txt')
    with open(train_file, 'w') as f:
        f.writelines(train_lines)
    
    # Write validation file
    val_file = input_file.replace('.txt', '_val.txt')
    with open(val_file, 'w') as f:
        f.writelines(val_lines)
    
    print(f"Dataset split:")
    print(f"Train: {len(train_lines)} images -> {train_file}")
    print(f"Validation: {len(val_lines)} images -> {val_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO dataset to FCOS format')
    parser.add_argument('--images_dir', required=True, help='Directory containing images')
    parser.add_argument('--labels_dir', required=True, help='Directory containing YOLO format labels')
    parser.add_argument('--output_file', required=True, help='Output file for FCOS format')
    parser.add_argument('--class_id', type=int, default=0, help='Class ID for license plates')
    parser.add_argument('--split', action='store_true', help='Split dataset into train/val')
    parser.add_argument('--train_ratio', type=float, default=0.85, help='Ratio for training set')
    
    args = parser.parse_args()
    
    # Convert dataset
    convert_yolo_to_fcos(
        args.images_dir, 
        args.labels_dir, 
        args.output_file, 
        args.class_id
    )
    
    # Split dataset if requested
    if args.split:
        split_dataset(args.output_file, args.train_ratio)

if __name__ == "__main__":
    main()
