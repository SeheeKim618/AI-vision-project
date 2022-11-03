# Deep Learning Model Implementation using Pytorch & Tensorflow
This repo is designed to be as easy as possible to implement deep learning models in pytorch and tensorflow. For study, you can see the material links I've attached.
* **Real-time Vehicle Detection** 
  * YOLOv5 [[github](https://github.com/ultralytics/yolov5)]
 
* **Classification on MNIST and CIFAR10/100**
  * CNN [[refer](https://cs231n.github.io/convolutional-networks/)]
  * LeNet [[paper](https://ieeexplore.ieee.org/document/726791)]
  * AlexNet [[paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)]
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

## AutoEncoders
This contains implementations of the following AutoEncoders:

* Standard autoencoder
* Convolutional autoencoder

### Results on MNIST
![conv_image_80](https://user-images.githubusercontent.com/62004821/199795488-171b0c65-ea53-43f8-8989-eb08095f29e9.png) ![conv_image_90](https://user-images.githubusercontent.com/62004821/199795501-b8cb2afb-43f7-4eed-a2cf-9785abfce732.png) ![KakaoTalk_Photo_2022-11-04-04-16-20](https://user-images.githubusercontent.com/76892271/199814032-f317a132-8f2b-42e4-93d1-a8ddf48c43fd.png)

## Documentation
You can find the API documentation on the pytorch website: https://pytorch.org/vision/stable/index.html
https://pytorch.org/docs/stable/index.html



