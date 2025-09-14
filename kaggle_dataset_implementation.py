#!/usr/bin/env python3
"""
Complete Implementation Guide for Kaggle Dataset
Step-by-step implementation for training, testing, and evaluation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil
import json
import time

class KaggleDatasetImplementation:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create all necessary directories"""
        directories = [
            self.output_path,
            self.output_path / "processed",
            self.output_path / "weights",
            self.output_path / "weights" / "object_detection",
            self.output_path / "weights" / "lprnet",
            self.output_path / "results",
            self.output_path / "results" / "object_detection",
            self.output_path / "results" / "lprnet",
            self.output_path / "lprnet_plates",
            self.output_path / "lprnet_plates" / "train",
            self.output_path / "lprnet_plates" / "test"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def step1_verify_dataset(self):
        """Step 1: Verify dataset structure and format"""
        print("\n" + "="*60)
        print("STEP 1: VERIFYING DATASET STRUCTURE")
        print("="*60)
        
        # Check if directories exist
        if not self.images_dir.exists():
            print(f"❌ Images directory not found: {self.images_dir}")
            return False
        if not self.labels_dir.exists():
            print(f"❌ Labels directory not found: {self.labels_dir}")
            return False
        
        # Count files
        image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        label_files = list(self.labels_dir.glob("*.txt"))
        
        print(f"📊 Dataset Statistics:")
        print(f"   Images: {len(image_files)}")
        print(f"   Labels: {len(label_files)}")
        
        # Check a sample label file
        if label_files:
            sample_label = label_files[0]
            print(f"\n📄 Sample label file: {sample_label.name}")
            try:
                with open(sample_label, 'r') as f:
                    content = f.read().strip()
                    print(f"   Content: {content}")
            except Exception as e:
                print(f"   Error reading label: {e}")
        
        # Check a sample image
        if image_files:
            sample_image = image_files[0]
            print(f"\n🖼️  Sample image: {sample_image.name}")
            try:
                import cv2
                img = cv2.imread(str(sample_image))
                if img is not None:
                    h, w = img.shape[:2]
                    print(f"   Dimensions: {w}x{h}")
                else:
                    print(f"   Error: Could not read image")
            except Exception as e:
                print(f"   Error reading image: {e}")
        
        print("✅ Dataset verification completed!")
        return True
    
    def step2_convert_dataset(self):
        """Step 2: Convert Kaggle dataset to FCOS format"""
        print("\n" + "="*60)
        print("STEP 2: CONVERTING DATASET TO FCOS FORMAT")
        print("="*60)
        
        convert_cmd = f"""python convert_dataset.py \
            --images_dir "{self.images_dir}" \
            --labels_dir "{self.labels_dir}" \
            --output_file "{self.output_path}/processed/fcos_dataset.txt" \
            --class_id 0 \
            --split \
            --train_ratio 0.85"""
        
        print(f"🔄 Running conversion command...")
        print(f"Command: {convert_cmd}")
        
        try:
            result = subprocess.run(convert_cmd, shell=True, check=True, capture_output=True, text=True)
            print("✅ Dataset conversion completed!")
            print("Output:", result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Conversion failed: {e.stderr}")
            return False
    
    def step3_prepare_lprnet_data(self):
        """Step 3: Prepare data for LPRNet training"""
        print("\n" + "="*60)
        print("STEP 3: PREPARING LPRNET DATA")
        print("="*60)
        
        extract_cmd = f"""python prepare_lprnet_dataset.py \
            --images_dir "{self.images_dir}" \
            --labels_dir "{self.labels_dir}" \
            --output_dir "{self.output_path}/lprnet_plates" \
            --target_size 94 24 \
            --create_structure \
            --train_ratio 0.85"""
        
        print(f"🔄 Extracting license plates...")
        print(f"Command: {extract_cmd}")
        
        try:
            result = subprocess.run(extract_cmd, shell=True, check=True, capture_output=True, text=True)
            print("✅ License plate extraction completed!")
            print("Output:", result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Extraction failed: {e.stderr}")
            return False
    
    def step4_train_object_detection(self):
        """Step 4: Train Object Detection Model (FCOS)"""
        print("\n" + "="*60)
        print("STEP 4: TRAINING OBJECT DETECTION MODEL (FCOS)")
        print("="*60)
        
        train_cmd = f"""python src/object_detection/train.py \
            --train_txt "{self.output_path}/processed/fcos_dataset_train.txt" \
            --batch_size 8 \
            --epochs 50 \
            --output_path "{self.output_path}/weights/object_detection" """
        
        print(f"🚀 Starting object detection training...")
        print(f"Command: {train_cmd}")
        print("⚠️  This will take several hours depending on your GPU...")
        
        # Ask for confirmation
        response = input("Do you want to start training now? (y/n): ")
        if response.lower() != 'y':
            print("⏸️  Training skipped. You can run it later with:")
            print(train_cmd)
            return True
        
        try:
            # Run training
            result = subprocess.run(train_cmd, shell=True, check=True)
            print("✅ Object detection training completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def step5_train_lprnet(self):
        """Step 5: Train LPRNet Character Recognition"""
        print("\n" + "="*60)
        print("STEP 5: TRAINING LPRNET CHARACTER RECOGNITION")
        print("="*60)
        
        train_cmd = f"""python src/License_Plate_Recognition/train_LPRNet.py \
            --train_img_dirs "{self.output_path}/lprnet_plates/train" \
            --test_img_dirs "{self.output_path}/lprnet_plates/test" \
            --max_epoch 200 \
            --train_batch_size 128 \
            --test_batch_size 128 \
            --save_folder "{self.output_path}/weights/lprnet/" """
        
        print(f"🚀 Starting LPRNet training...")
        print(f"Command: {train_cmd}")
        print("⚠️  This will take several hours depending on your GPU...")
        
        # Ask for confirmation
        response = input("Do you want to start training now? (y/n): ")
        if response.lower() != 'y':
            print("⏸️  Training skipped. You can run it later with:")
            print(train_cmd)
            return True
        
        try:
            # Run training
            result = subprocess.run(train_cmd, shell=True, check=True)
            print("✅ LPRNet training completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def step6_evaluation_metrics(self):
        """Step 6: Set up evaluation metrics"""
        print("\n" + "="*60)
        print("STEP 6: SETTING UP EVALUATION METRICS")
        print("="*60)
        
        # Create evaluation script for object detection
        eval_od_script = f"""#!/usr/bin/env python3
# Object Detection Evaluation Script

import sys
sys.path.append('src/object_detection')

from eval import validate_one_epoch
from model.fcos import FCOSDetector
import torch
from dataloader.custom_dataset import YoloDataset
from torch.utils.data import DataLoader

def evaluate_object_detection():
    # Load model
    model = FCOSDetector(mode="inference").cuda()
    model.load_state_dict(torch.load("{self.output_path}/weights/object_detection/best_od.pth"))
    model.eval()
    
    # Load validation dataset
    val_dataset = YoloDataset("{self.output_path}/processed/fcos_dataset_val.txt")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=val_dataset.collate_fn)
    
    # Run evaluation
    print("Running object detection evaluation...")
    # Add your evaluation code here
    
if __name__ == "__main__":
    evaluate_object_detection()
"""
        
        with open(f"{self.output_path}/results/evaluate_object_detection.py", "w") as f:
            f.write(eval_od_script)
        
        # Create evaluation script for LPRNet
        eval_lpr_script = f"""#!/usr/bin/env python3
# LPRNet Evaluation Script

import sys
sys.path.append('src/License_Plate_Recognition')

from test_LPRNet import Greedy_Decode_Eval
from model.LPRNet import build_lprnet
import torch

def evaluate_lprnet():
    # Load model
    lprnet = build_lprnet(lpr_max_len=16, phase=False, class_num=37, dropout_rate=0.5)
    lprnet.load_state_dict(torch.load("{self.output_path}/weights/lprnet/best_lprnet.pth"))
    lprnet.eval()
    
    # Load test dataset
    from data.load_data import LPRDataLoader
    test_dataset = LPRDataLoader(["{self.output_path}/lprnet_plates/test"], (94, 24), 16)
    
    # Run evaluation
    print("Running LPRNet evaluation...")
    Greedy_Decode_Eval(lprnet, test_dataset, None)
    
if __name__ == "__main__":
    evaluate_lprnet()
"""
        
        with open(f"{self.output_path}/results/evaluate_lprnet.py", "w") as f:
            f.write(eval_lpr_script)
        
        print("✅ Evaluation scripts created!")
        print(f"   - {self.output_path}/results/evaluate_object_detection.py")
        print(f"   - {self.output_path}/results/evaluate_lprnet.py")
        
        return True
    
    def step7_testing_pipeline(self):
        """Step 7: Create complete testing pipeline"""
        print("\n" + "="*60)
        print("STEP 7: CREATING TESTING PIPELINE")
        print("="*60)
        
        # Create inference script
        inference_script = f"""#!/usr/bin/env python3
# Complete Inference Pipeline

import cv2
import torch
import numpy as np
import sys
sys.path.append('src/object_detection')
sys.path.append('src/License_Plate_Recognition')

from model.fcos import FCOSDetector
from model.LPRNet import build_lprnet

class IndianLPRPipeline:
    def __init__(self, od_weights_path, lpr_weights_path):
        # Load object detection model
        self.od_model = FCOSDetector(mode="inference").cuda()
        self.od_model.load_state_dict(torch.load(od_weights_path))
        self.od_model.eval()
        
        # Load LPRNet model
        self.lpr_model = build_lprnet(lpr_max_len=16, phase=False, class_num=37, dropout_rate=0.5)
        self.lpr_model.load_state_dict(torch.load(lpr_weights_path))
        self.lpr_model.eval()
        
    def detect_license_plates(self, image):
        # Preprocess image
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).cuda()
        img_tensor = img_tensor / 255.0
        
        # Run object detection
        with torch.no_grad():
            scores, classes, boxes = self.od_model(img_tensor)
        
        return scores, classes, boxes
    
    def recognize_text(self, plate_image):
        # Preprocess plate image
        plate_resized = cv2.resize(plate_image, (94, 24))
        plate_tensor = torch.from_numpy(plate_resized).permute(2, 0, 1).float().unsqueeze(0)
        plate_tensor = (plate_tensor - 127.5) * 0.0078125
        
        # Run LPRNet
        with torch.no_grad():
            logits = self.lpr_model(plate_tensor)
            # Add decoding logic here
        
        return "PLATE_TEXT"  # Placeholder
    
    def process_image(self, image_path):
        # Load image
        image = cv2.imread(image_path)
        
        # Detect license plates
        scores, classes, boxes = self.detect_license_plates(image)
        
        results = []
        for i in range(len(scores)):
            if scores[i] > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = boxes[i]
                plate_img = image[y1:y2, x1:x2]
                text = self.recognize_text(plate_img)
                results.append({{'bbox': [x1, y1, x2, y2], 'text': text, 'confidence': scores[i]}})
        
        return results

if __name__ == "__main__":
    pipeline = IndianLPRPipeline(
        "{self.output_path}/weights/object_detection/best_od.pth",
        "{self.output_path}/weights/lprnet/best_lprnet.pth"
    )
    
    # Test on sample image
    results = pipeline.process_image("demo_images/20201031_133155_3220.jpg")
    print("Results:", results)
"""
        
        with open(f"{self.output_path}/results/complete_inference.py", "w") as f:
            f.write(inference_script)
        
        print("✅ Testing pipeline created!")
        print(f"   - {self.output_path}/results/complete_inference.py")
        
        return True
    
    def run_all_steps(self):
        """Run all implementation steps"""
        print("🚀 STARTING COMPLETE KAGGLE DATASET IMPLEMENTATION")
        print("="*80)
        
        steps = [
            ("Dataset Verification", self.step1_verify_dataset),
            ("Dataset Conversion", self.step2_convert_dataset),
            ("LPRNet Data Preparation", self.step3_prepare_lprnet_data),
            ("Object Detection Training", self.step4_train_object_detection),
            ("LPRNet Training", self.step5_train_lprnet),
            ("Evaluation Setup", self.step6_evaluation_metrics),
            ("Testing Pipeline", self.step7_testing_pipeline)
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔄 Executing: {step_name}")
            try:
                success = step_func()
                if not success:
                    print(f"❌ {step_name} failed!")
                    return False
            except Exception as e:
                print(f"❌ {step_name} failed with error: {e}")
                return False
        
        print("\n" + "="*80)
        print("🎉 COMPLETE IMPLEMENTATION FINISHED!")
        print("="*80)
        print(f"📁 All results saved in: {self.output_path}")
        print("\n📋 Next steps:")
        print("1. Check training logs in the weights directories")
        print("2. Run evaluation scripts to test performance")
        print("3. Use the complete inference pipeline for testing")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Complete Kaggle Dataset Implementation')
    parser.add_argument('--dataset_path', required=True, help='Path to your Kaggle dataset (containing images/ and labels/ folders)')
    parser.add_argument('--output_path', default='./kaggle_implementation', help='Output path for processed data and results')
    parser.add_argument('--step', type=int, help='Run specific step (1-7)')
    
    args = parser.parse_args()
    
    implementation = KaggleDatasetImplementation(args.dataset_path, args.output_path)
    
    if args.step:
        # Run specific step
        steps = {
            1: implementation.step1_verify_dataset,
            2: implementation.step2_convert_dataset,
            3: implementation.step3_prepare_lprnet_data,
            4: implementation.step4_train_object_detection,
            5: implementation.step5_train_lprnet,
            6: implementation.step6_evaluation_metrics,
            7: implementation.step7_testing_pipeline
        }
        
        if args.step in steps:
            steps[args.step]()
        else:
            print(f"Invalid step number: {args.step}. Valid steps: 1-7")
    else:
        # Run all steps
        implementation.run_all_steps()

if __name__ == "__main__":
    main()
