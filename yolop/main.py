import pyzed.sl as sl
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from utility import Yolop
import torch
import pandas as pd

# initialize the model
model=Yolop()
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(model.model.names))]


# Create a ZED camera object
zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
init_params.camera_fps = 60  # Set fps at 60


err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

image=sl.Mat(1280,720,sl.MAT_TYPE.U8_C4) #camera width and hieght in zed image array
ocv=image.get_data()#converts mat(zed image format) to numpy array


start=time.time() # used to calculate the total latency of the algo
frame=0 # used to calculate FPS

return_latency=True # set to false if latency statistics are nor needed
latency_results=[]

# read the test image
test=cv2.imread('test.jpg')
cv2.imshow('test', test)

with torch.inference_mode():
    if model.device != 'cpu':
        model(torch.zeros((1, 3, 640, 640), device=model.device)) # as per documentation of the yolop model
    model.eval()
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            ocv = image.get_data()
            img_l = np.copy(ocv[:, :, :3])  # Extract RGB channels
            frame += 1


            # pass the image through the model
            result = model.detect(img_l, visualize=False, colors=colors, return_latency=return_latency)
            latency_results.append(result[-2])
            # cv2.imshow('Left Image', img_l)
            # cv2.imshow('Bounded Image', result[-1])  # Assuming result[-1] is the processed image

            if frame == 10:
                print("FPS:", 10 / (time.time() - start))
                frame = 0
                start = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    zed.close()

if return_latency:
    latency_results=pd.DataFrame(latency_results)
    ax=plt.axes()
    for col in latency_results.columns:
        if col=='total' or col=='model_pass':
            continue
        ax.plot(latency_results[col], label=col)
    ax.set_title('Latency')
    ax.set_ylabel('Time(s)')
    ax.set_xlabel('Frame')
    ax.legend()
    plt.gcf().set_size_inches(10, 10)
    plt.show()
    agrregate_results=latency_results.aggregate(func=['mean', 'min', 'max'], axis=0)
    print(agrregate_results)
    agrregate_results.to_csv('aggregate_results_rtx_yolop.csv')
