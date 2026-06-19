import os
import random
from collections import defaultdict

def create_stratified_split(xml_files_list, train_ratio=0.8, seed=42):
    """
    Groups files by manipulation type and splits them 80/20 individually.
    Prevents OOD manipulation tasks from exclusively appearing in the validation set.
    """
    random.seed(seed)
    
    # 1. Group files by their manipulation category
    categories = defaultdict(list)
    
    for filename in xml_files_list:
        # Example filename: "01_2PM_2PCM_01.xml"
        base_name = filename.replace('.xml', '') 
        parts = base_name.split('_')
        
        # The manipulation type is everything between the first and last numbers
        # Extracted: "2PM_2PCM"
        manipulation_type = "_".join(parts[1:-1]) 
        
        categories[manipulation_type].append(filename)
        
    train_files = []
    val_files = []
    
    # 2. Iterate through each category and split 80/20
    print("--- Stratified Split Summary ---")
    for manip_type, files in categories.items():
        # Shuffle to ensure random trial allocation
        random.shuffle(files)
        
        split_idx = int(len(files) * train_ratio)
        train_split = files[:split_idx]
        val_split = files[split_idx:]
        
        train_files.extend(train_split)
        val_files.extend(val_split)
        
        print(f"[{manip_type}]: {len(train_split)} Train | {len(val_split)} Val")
        
    print(f"\nTotal Train: {len(train_files)} | Total Val: {len(val_files)}")
    print("--------------------------------")
    
    return train_files, val_files