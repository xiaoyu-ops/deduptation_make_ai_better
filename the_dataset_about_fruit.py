from datasets import load_dataset
from PIL import Image

ds = load_dataset("microsoft/cats_vs_dogs")
ds.save_to_disk("data/cats_vs_dogs")
print(ds["train"][0].keys())
print(len(ds["train"]))#23410张照片 
image = ds["train"][12000]["image"]
image.show()