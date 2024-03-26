import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation,Mask2FormerImageProcessor



import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
img = cv2.imread("testim.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# load Mask2Former fine-tuned on Mapillary Vistas panoptic segmentation
#processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")
processor =  Mask2FormerImageProcessor(ignore_index=255)
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")
model.config.ignore_value= 255
# load Mask2Former fine-tuned on Mapillary Vistas panoptic segmentation
#processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")
processor =  Mask2FormerImageProcessor(ignore_index=255)
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")
model.config.ignore_value= 255
#%%time
with torch.no_grad():
    outputs = model(**inputs)
def draw_panoptic_segmentation(segmentation, segments_info):
    # get the used color map
    viridis = cm.get_cmap('viridis',8)
    fig, ax = plt.subplots()
    
    instances_counter = defaultdict(int)
    handles = []
    
    ax.imshow(segmentation)
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        #color = (sum(cmapseg[segment_label])*100)//1
        #img = Image.fromarray(np.array(segmentation), 'RGB')
        #segmentation[segmentation==segment_id] =color
        #segmentation = np.array(img)
        #segmentation[:,:,0] =np.array(result['segmentation'])
        #segmentation = np.copy(segmentation)
        
        #segmentation[segmentation[:,:,0]==segment_id]=color*255
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        print(segment_label_id, segment_label,segment_id)
        instances_counter[segment_label] += 1
        #color=mcolors.to_rgb(color)
        handles.append(mpatches.Patch(color=viridis(segment_id), label=label))
    #segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB) 
     #,cmap='viridis')  
    #ax.legend(handles=handles)
#result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.shape[:-1]])[0]
draw_panoptic_segmentation(**result)
