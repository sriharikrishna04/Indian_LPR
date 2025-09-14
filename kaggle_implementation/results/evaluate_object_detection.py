#!/usr/bin/env python3
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
    model.load_state_dict(torch.load("kaggle_implementation/weights/object_detection/best_od.pth"))
    model.eval()
    
    # Load validation dataset
    val_dataset = YoloDataset("kaggle_implementation/processed/fcos_dataset_val.txt")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=val_dataset.collate_fn)
    
    # Run evaluation
    print("Running object detection evaluation...")
    # Add your evaluation code here
    
if __name__ == "__main__":
    evaluate_object_detection()
