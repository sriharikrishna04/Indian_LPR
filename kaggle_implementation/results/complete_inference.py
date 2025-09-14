#!/usr/bin/env python3
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
                results.append({'bbox': [x1, y1, x2, y2], 'text': text, 'confidence': scores[i]})
        
        return results

if __name__ == "__main__":
    pipeline = IndianLPRPipeline(
        "kaggle_implementation/weights/object_detection/best_od.pth",
        "kaggle_implementation/weights/lprnet/best_lprnet.pth"
    )
    
    # Test on sample image
    results = pipeline.process_image("demo_images/20201031_133155_3220.jpg")
    print("Results:", results)
