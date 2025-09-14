#!/usr/bin/env python3
# LPRNet Evaluation Script

import sys
sys.path.append('src/License_Plate_Recognition')

from test_LPRNet import Greedy_Decode_Eval
from model.LPRNet import build_lprnet
import torch

def evaluate_lprnet():
    # Load model
    lprnet = build_lprnet(lpr_max_len=16, phase=False, class_num=37, dropout_rate=0.5)
    lprnet.load_state_dict(torch.load("kaggle_implementation/weights/lprnet/best_lprnet.pth"))
    lprnet.eval()
    
    # Load test dataset
    from data.load_data import LPRDataLoader
    test_dataset = LPRDataLoader(["kaggle_implementation/lprnet_plates/test"], (94, 24), 16)
    
    # Run evaluation
    print("Running LPRNet evaluation...")
    Greedy_Decode_Eval(lprnet, test_dataset, None)
    
if __name__ == "__main__":
    evaluate_lprnet()
