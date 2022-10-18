from transformers import BeitFeatureExtractor, BeitModel
import torch.nn as nn
from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
# feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-512',cache_dir="./temp")
feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224',cache_dir="./temp")
model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224',cache_dir="./temp")
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

avgpool = nn.AdaptiveAvgPool2d((28, 768))
temp =  nn.Linear(768,768)

tzz = 1