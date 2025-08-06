import random
from xml.etree.ElementInclude import DEFAULT_MAX_INCLUSION_DEPTH

from datasets import load_dataset

DonutDataset = load_dataset("imagefolder", data_dir="data/img", split="train")

random_sample = random.randint(0, len(DonutDataset))

print(f"Random sample is {random_sample}")
print(f"OCR text is {DonutDataset[random_sample]['text']}")
#     OCR text is {"company": "LIM SENG THO HARDWARE TRADING", "date": "29/12/2017", "address": "NO 7, SIMPANG OFF BATU VILLAGE, JALAN IPOH BATU 5, 51200 KUALA LUMPUR MALAYSIA", "total": "6.00"}