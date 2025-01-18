import pyzed.sl as sl
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from elements.yolo import OBJ_DETECTION


Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush' ]

Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('weights/yolov5s.pt', Object_classes)


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

visualize=True # to visualize the results by annotating boxes on the image
return_latency=True # to capture latency statistics
latency_results=[]

test=cv2.imread('test.jpg')
Object_detector.detect(test)
cv2.imshow('test', test)

while True:
    
    if zed.grab()==sl.ERROR_CODE.SUCCESS:
        "grabs image frame from zed"
        zed.retrieve_image(image,sl.VIEW.LEFT)
        ocv=image.get_data()
        img_l=np.copy(ocv[:, :, :3])
        frame+=1
        model_start=time.time()
        objs=Object_detector.detect(img_l)
        latency_results.append(time.time()-model_start)
        if visualize:
            cv2.imshow("img",img_l)
            
            for obj in objs:
                label=obj['label']
                score = obj['score']
                [(xmin,ymin),(xmax,ymax)] = obj['bbox']
                color = Object_colors[Object_classes.index(label)]
                img_l = cv2.rectangle(img_l, (xmin,ymin), (xmax,ymax), color, 2) 
                img_l = cv2.putText(img_l, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)
            
            cv2.imshow("bounded_image",img_l)

        if frame==10:
            print("fps",10/(time.time()-start))
            frame=0
            start=time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows
            zed.close()
            break




if return_latency:
    latency_results=pd.Series(latency_results)
    agrregate_results=latency_results.aggregate(func=['mean', 'min', 'max'], axis=0)
    print(agrregate_results)
    agrregate_results.to_csv('aggregate_results_rtx_yolov5.csv')