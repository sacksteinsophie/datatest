import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches
from matplotlib import cm

def process_panoptic_segmentation(image_path, save_path):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize Mask2Former processor and model
    processor = Mask2FormerImageProcessor(ignore_index=255)
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")
    model.config.ignore_value = 255
    
    # Perform inference
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        outputs = model(**inputs)
    
    # Process segmentation result
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[img.shape[:-1]])[0]
    
    # Save segmentation mask
    segmentation_mask = result['segmentation']
    segmentation_mask.save(save_path)

# Example usage:
# process_panoptic_segmentation("testim.png", "segmentation_mask.png")
