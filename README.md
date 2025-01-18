# Winter Project
## IIT Madras, Joint MSc Data Science and Artificial Intelligence
## December 2024 - January 2025
### Object Detection in Scaled Down Autonomous Vehicles

#### Abstract
Object detection is a fundamental component of autonomous vehicle
systems, enabling real-time perception and navigation in dynamic envi-
ronments. This project focuses on implementing and comparing two deep
learning algorithms, YOLOv5 and YOLOP, for object detection tasks in
the context of autonomous driving. YOLOv5 is known for its high detec-
tion accuracy and speed, while YOLOP offers an integrated approach by
combining object detection, lane detection, and road segmentation.
The existing GitHub implementations of these algorithms were stream-
lined to process video frames in real-time. The optimized pipeline was
tested on two different GPU platforms: the Jetson Nano and the GeForce
RTX 3080 Ti; to evaluate the algorithms’ performance in terms of speed.
Performance analysis revealed that YOLOv5 achieved a frame rate of
16 FPS on the Jetson Nano and 50 FPS on the GeForce RTX 3080 Ti,
outperforming YOLOP, which achieved 3 FPS and 25 FPS, respectively,
on the same hardware. These results highlight YOLOv5’s superior effi-
ciency and scalability, particularly for real-time applications in resource-
constrained environments. While YOLOP demonstrated its capability to
perform multiple perception tasks simultaneously, its lower frame rates
suggest a trade-off between functionality and performance.

#### Acknowledgment
I would like to express my gratitude to my guide, Dr. Ramkrishna Pasumarthy,
for his invaluable guidance and encouragement throughout the project. His
expertise and constructive feedback have been instrumental in the successful
completion of this work. I am also deeply thankful to Ms.Vasumathi R, whose
insightful suggestions and assistance greatly contributed to achieving the re-
sults in this project. Their support has been an essential part of my learning
experience.


#### Code Reference
For code document, please refer to the `code_doc.md` file.