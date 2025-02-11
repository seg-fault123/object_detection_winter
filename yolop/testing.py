import torch
import cv2
import numpy as np
from utility import Yolop
import matplotlib.pyplot as plt

image=cv2.imread('./source_images/test.jpg')
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis(False)
plt.show()


model=Yolop()
with torch.inference_mode():
    if model.device != 'cpu':
        model(torch.zeros((1, 3, 640, 640), device=model.device))
    model.eval()
    colors = [[0, 255, 255]]
    # colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(model.model.names))]
    result=model.detect(image, visualize=True, colors=colors, return_latency=True)

    ## the annotated image
    plt.imshow(result[-1])
    plt.axis(False)
    plt.show()
    
    ## the road detection mask
    plt.imshow(result[1], cmap='gray')
    plt.axis(False)
    plt.show()

    ## the lane detection mask
    plt.imshow(result[2], cmap='gray')
    plt.axis(False)
    plt.show()
    


