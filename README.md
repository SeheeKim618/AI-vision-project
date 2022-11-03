# Deep Learning Model Implementation using Pytorch & Tensorflow
This repo is designed to be as easy as possible to implement deep learning models in pytorch and tensorflow. For study, you can see the material links I've attached.
* **Real-time Vehicle Detection** 
  * YOLOv5 [[github](https://github.com/ultralytics/yolov5)]
 
* **Classification on MNIST and CIFAR10/100**
  * CNN [[refer](https://cs231n.github.io/convolutional-networks/)]
  * LeNet
  * AlexNet
  * VGGNet-11/16 [[paper](https://arxiv.org/pdf/1409.1556v6.pdf)]
  * ResNet-18/34/50 [[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)]
  * GoogleNet [[paper](https://arxiv.org/pdf/1409.4842.pdf)]
  
 * **AutoEncoder on MNIST**
   * AutoEncoder study material in Korean [[youtube](https://www.youtube.com/watch?v=o_peo6U7IRM)]
  
 * **Reinforcement Learning(A2C) on Atari**
   * Material in Korean [[link](https://ropiens.tistory.com/153)][[link](https://ropiens.tistory.com/163)]

## Real-time Vehicle Detection using YOLOv5s
The code is to train a custom dataset for YOLOv5 and fine-tune a object detection model. The model will be ready for real-time object detection.

### Dataset 

The dataset used for training here is **Vehicles-OpenImages Dataset** in roboflow. [[link](https://public.roboflow.com/object-detection/vehicles-openimages)]
 <br/>
For inference, you can use your own vehicle dataset.

### Results on real-time video
<img width="60%" src="https://user-images.githubusercontent.com/76892271/199302043-2e9540c6-e1eb-49eb-a82b-6e7737845265.gif"/>
<img width="60%" src="https://user-images.githubusercontent.com/76892271/199309422-5a1e43f3-5249-4c09-88cf-801ba789bd72.gif"/>

## Documentation
You can find the API documentation on the pytorch website: https://pytorch.org/vision/stable/index.html
https://pytorch.org/docs/stable/index.html



