import os
import json
from pathlib import Path
import shutil
from datasets import load_dataset
 
# define paths
base_path = Path("data")
metadata_path = base_path.joinpath("key")
image_path = base_path.joinpath("img")
 
# Load dataset
dataset = load_dataset("imagefolder", data_dir=image_path, split="train")
 
print(f"Dataset has {len(dataset)} images")
print(f"Dataset features are: {dataset.features.keys()}")