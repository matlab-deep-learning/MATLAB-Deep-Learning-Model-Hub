# MATLAB Deep Learning Model Hub

Discover pretrained models for deep learning in MATLAB.

## Models <a name="Models"/>

### Computer Vision
* [Image Classification](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#image-classification-)
* [Object Detection](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#object-detection-)
* [Semantic Segmentation](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#semantic-segmentation-)
* [Instance Segmentation](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#instance-segmentation-)
* [Image Translation](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#image-translation-)
* [Pose Estimation](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#pose-estimation-)
* [3D Reconstruction](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#3d-reconstruction-)
* [Video Classification](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#video-classification-)
* [Text Detection & Recognition](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#text-detection-and-recognition-)

### Natural Language Processing
* [Transformers](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#transformers-text-)

### Audio
* [Audio Embeddings](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#audio-embeddings-)
* [Sound Classification](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#application-specific-audio-models)
* [Pitch Estimation](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#application-specific-audio-models)
* [Speech to Text](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#speech-to-text-)

### Lidar 
* [Point Cloud Classification](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#lidar-)
* [Point Cloud Segmentation](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#lidar-)
* [Point Cloud Object Detection](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#lidar-)

### Robotics
* [Manipulator Motion Planning](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#manipulator-motion-planning-)
* [Path Planning with Motion Planning Networks](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#path-planning-with-motion-planning-networks-)

## Image Classification <a name="ImageClassification"/>

Pretrained image classification networks have already learned to extract powerful and informative features from natural images. Use them as a starting point to learn a new task using transfer learning. 

Inputs are RGB images, the output is the predicted label and score:

![](Images/classification_workflow.png)

These networks have been trained on more than a million images and can classify images into 1000 object categories. 

**Models available in MATLAB:**

**Note 1: Since R2024a, please use the [imagePretrainedNetwork](https://www.mathworks.com/help/deeplearning/ref/imagepretrainednetwork.html) function instead and specify the pretrained model**.
For example, use the following code to access googlenet:
```
[net, classes] = imagePretrainedNetwork("googlenet");
```

| Network |   Size (MB)  | Classes | Accuracy % | Location | 
| ------------- | ------------- | ------------- |  ------------- |  ------------- | 
| [googlenet](https://www.mathworks.com/help/deeplearning/ref/googlenet.html)<sup>1<sup> | 27| 1000| 66.25 |[Doc](https://www.mathworks.com/help/deeplearning/ref/googlenet.html) <br />[GitHub](https://github.com/matlab-deep-learning/googlenet) |
| [squeezenet](https://www.mathworks.com/help/deeplearning/ref/squeezenet.html)<sup>1<sup>   | 5.2|  1000| 55.16 |[Doc](https://www.mathworks.com/help/deeplearning/ref/squeezenet.html) |
| [alexnet](https://www.mathworks.com/help/deeplearning/ref/alexnet.html)<sup>1<sup>  |  227|  1000|54.10|[Doc](https://www.mathworks.com/help/deeplearning/ref/alexnet.html) |
| [resnet18](https://www.mathworks.com/help/deeplearning/ref/resnet18.html)<sup>1<sup>   |  44|  1000|69.49|[Doc](https://www.mathworks.com/help/deeplearning/ref/resnet18.html) <br />[GitHub](https://github.com/matlab-deep-learning/resnet-18) |
| [resnet50](https://www.mathworks.com/help/deeplearning/ref/resnet50.html)<sup>1<sup>   |  96|  1000|74.46|[Doc](https://www.mathworks.com/help/deeplearning/ref/resnet50.html) <br />[GitHub](https://github.com/matlab-deep-learning/resnet-50) |
| [resnet101](https://www.mathworks.com/help/deeplearning/ref/resnet101.html)<sup>1<sup>   |  167|  1000|75.96|[Doc](https://www.mathworks.com/help/deeplearning/ref/resnet101.html) <br />[GitHub](https://github.com/matlab-deep-learning/resnet-101) |
| [mobilenetv2](https://www.mathworks.com/help/deeplearning/ref/mobilenetv2.html)<sup>1<sup>    | 13| 1000| 70.44|[Doc](https://www.mathworks.com/help/deeplearning/ref/mobilenetv2.html) <br />[GitHub](https://github.com/matlab-deep-learning/mobilenet-v2) |
| [vgg16](https://www.mathworks.com/help/deeplearning/ref/vgg16.html)<sup>1<sup>    |  515|  1000|70.29|[Doc](https://www.mathworks.com/help/deeplearning/ref/vgg16.html) |
| [vgg19](https://www.mathworks.com/help/deeplearning/ref/vgg19.html)<sup>1<sup>    |  535|  1000|70.42|[Doc](https://www.mathworks.com/help/deeplearning/ref/vgg19.html) |
| [inceptionv3](https://www.mathworks.com/help/deeplearning/ref/inceptionv3.html)<sup>1<sup> | 89|  1000|77.07|[Doc](https://www.mathworks.com/help/deeplearning/ref/inceptionv3.html) |
| [inceptionresnetv2](https://www.mathworks.com/help/deeplearning/ref/inceptionresnetv2.html)<sup>1<sup>  | 209|  1000|79.62|[Doc](https://www.mathworks.com/help/deeplearning/ref/inceptionresnetv2.html) |
| [xception](https://www.mathworks.com/help/deeplearning/ref/xception.html)<sup>1<sup>   |  85|  1000|78.20|[Doc](https://www.mathworks.com/help/deeplearning/ref/xception.html) |
| [darknet19](https://www.mathworks.com/help/deeplearning/ref/darknet19.html)<sup>1<sup>   |  78| 1000|74.00|[Doc](https://www.mathworks.com/help/deeplearning/ref/darknet19.html) |
| [darknet53](https://www.mathworks.com/help/deeplearning/ref/darknet53.html)<sup>1<sup>    |  155|  1000|76.46|[Doc](https://www.mathworks.com/help/deeplearning/ref/darknet53.html) |
| [densenet201](https://www.mathworks.com/help/deeplearning/ref/densenet201.html)<sup>1<sup>    | 77| 1000| 75.85|[Doc](https://www.mathworks.com/help/deeplearning/ref/densenet201.html) |
| [shufflenet](https://www.mathworks.com/help/deeplearning/ref/shufflenet.html)<sup>1<sup>   | 5.4|  1000|63.73|[Doc](https://www.mathworks.com/help/deeplearning/ref/shufflenet.html) |
| [nasnetmobile](https://www.mathworks.com/help/deeplearning/ref/nasnetmobile.html)<sup>1<sup>    | 20|  1000|73.41|[Doc](https://www.mathworks.com/help/deeplearning/ref/nasnetmobile.html) |
| [nasnetlarge](https://www.mathworks.com/help/deeplearning/ref/nasnetlarge.html)<sup>1<sup>    |  332| 1000|81.83|[Doc](https://www.mathworks.com/help/deeplearning/ref/nasnetlarge.html) |
| [efficientnetb0](https://www.mathworks.com/help/deeplearning/ref/efficientnetb0.html)<sup>1<sup>   | 20|  1000|74.72|[Doc](https://www.mathworks.com/help/deeplearning/ref/efficientnetb0.html) |
| [ConvMixer](https://github.com/matlab-deep-learning/convmixer-patches-are-all-you-need)   | 7.7|  10|-|[GitHub](https://github.com/matlab-deep-learning/convmixer-patches-are-all-you-need) |
| [Vison Transformer](https://www.mathworks.com/help/vision/ref/visiontransformer.html)   | Large-16 - 1100<br /> Base-16 - 331.4<br /> Small-16 - 84.7<br /> Tiny-16 - 22.2|  1000|Large-16 - 85.59<br /> Base-16 - 85.49<br /> Small-16 - 83.73<br /> Tiny-16 - 78.22|[Doc](https://www.mathworks.com/help/vision/ref/visiontransformer.html) |

**Tips for selecting a model**

Pretrained networks have different characteristics that matter when choosing a network to apply to your problem. The most important characteristics are network accuracy, speed, and size. Choosing a network is generally a tradeoff between these characteristics. The following figure highlights these tradeoffs:

![](Images/pretrained.png)
Figure. Comparing image classification model accuracy, speed and size.

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Object Detection <a name="ObjectDetection"/>

Object detection is a computer vision technique used for locating instances of objects in images or videos. When humans look at images or video, we can recognize and locate objects of interest within a matter of moments. The goal of object detection is to replicate this intelligence using a computer.

Inputs are RGB images, the output is the predicted label, bounding box and score:

![](Images/objectdetection_workflow.png)

These networks have been trained to detect 80 objects classes from the COCO dataset. These models are suitable for training a custom object detector using transfer learning.

| Network  | Network variants |  Size (MB) | Mean Average Precision (mAP) |Object Classes | Location |
| ------------- | ------------- |------------ | ------------- |------------ |------------ |
| [EfficientDet-D0](https://github.com/matlab-deep-learning/pretrained-efficientdet-d0) | efficientnet |15.9 | 33.7 |80 |[GitHub](https://github.com/matlab-deep-learning/pretrained-efficientdet-d0) | 
| [YOLO v8](https://github.com/matlab-deep-learning/Pretrained-YOLOv8-Network-For-Object-Detection) | yolo8n<br />yolo8s<br />yolo8m<br />yolo8l<br />yolo8x | 10.7 <br /> 37.2<br />85.4 <br />143.3<br />222.7 | 37.3<br />44.9<br />50.2<br />52.9<br />53.9|80 |[GitHub](https://github.com/matlab-deep-learning/Pretrained-YOLOv8-Network-For-Object-Detection)| 
| [YOLOX](https://www.mathworks.com/help/vision/ref/yoloxobjectdetector.html) | YoloX-s<br />YoloX-m<br />YoloX-l | 32 <br /> 90.2<br />192.9 | 39.8 <br />45.9<br />48.6|80 |[Doc](https://www.mathworks.com/help/vision/ref/yoloxobjectdetector.html)<br />[GitHub](https://github.com/matlab-deep-learning/Pretrained-YOLOX-Network-For-Object-Detection)| 
| [YOLO v4](https://www.mathworks.com/help/vision/ref/yolov4objectdetector.html) | yolov4-coco <br /> yolov4-tiny-coco| 229 <br /> 21.5 | 44.2 <br />19.7|80 |[Doc](https://www.mathworks.com/help/vision/ref/yolov4objectdetector.html)<br />[GitHub](https://github.com/matlab-deep-learning/pretrained-yolo-v4)| 
| [YOLO v3](https://www.mathworks.com/help/vision/ref/yolov3objectdetector.html)| darknet53-coco <br /> tiny-yolov3-coco | 220.4 <br /> 31.5 | 34.4 <br /> 9.3 |80 |[Doc](https://www.mathworks.com/help/vision/ref/yolov3objectdetector.html) |
| [YOLO v2](https://www.mathworks.com/help/vision/ref/yolov2objectdetector.html)   | darknet19-COCO <br />tiny-yolo_v2-coco|181 <br /> 40 | 28.7 <br /> 10.5 |80 |[Doc](https://www.mathworks.com/help/vision/ref/yolov2objectdetector.html)<br />[GitHub](https://github.com/matlab-deep-learning/Object-Detection-Using-Pretrained-YOLO-v2)|

**Tips for selecting a model**

Pretrained object detectors have different characteristics that matter when choosing a network to apply to your problem. The most important characteristics are mean average precision (mAP), speed, and size. Choosing a network is generally a tradeoff between these characteristics.

**Application Specific Object Detectors**

These networks have been trained to detect specific objects for a given application.
| Network  | Application |Size (MB) |Location |Example Output |
| ------------- | ------------- | ------------- |------------- |------------- |
| [Spatial-CNN](https://github.com/matlab-deep-learning/pretrained-spatial-CNN)   | Lane detection | 74 |[GitHub ](https://github.com/matlab-deep-learning/pretrained-spatial-CNN)|<img src="Images/lanedetection.jpg" width=150>|
| [RESA](https://github.com/matlab-deep-learning/Pretrained-RESA-Network-For-Road-Boundary-Detection)   | Road Boundary detection | 95 |[GitHub ](https://github.com/matlab-deep-learning/Pretrained-RESA-Network-For-Road-Boundary-Detection)|<img src="Images/road_boundary.png" width=150>|
| [Single Shot Detector (SSD)](https://www.mathworks.com/help/vision/ug/object-detection-using-single-shot-detector.html)   | Vehicle detection | 44 |[Doc ](https://www.mathworks.com/help/vision/ug/object-detection-using-single-shot-detector.html)|<img src="Images/ObjectDetectionUsingSSD.png" width=150>|
| [Faster R-CNN](https://www.mathworks.com/help/vision/ug/object-detection-using-faster-r-cnn-deep-learning.html)   | Vehicle detection | 118 |[Doc](https://www.mathworks.com/help/vision/ug/object-detection-using-faster-r-cnn-deep-learning.html)|<img src="Images/faster_rcnn.png" width=150>|

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Semantic Segmentation <a name="SemanticSegmentation"/>

Segmentation is essential for image analysis tasks. Semantic segmentation describes the process of associating each pixel of an image with a class label, (such as flower, person, road, sky, ocean, or car). 

Inputs are RGB images, outputs are pixel classifications (semantic maps).
<img src="Images/semanticseg.png" class="center"> 

This network has been trained to detect 20 objects classes from the PASCAL VOC dataset:

| Network  | Size (MB)| Mean Accuracy | Object Classes| Location | 
| ------------- | ------------- |------------- |------------- |------------- |
| [DeepLabv3+](https://github.com/matlab-deep-learning/pretrained-deeplabv3plus)   | 209 | 0.87 | 20 | [GitHub](https://github.com/matlab-deep-learning/pretrained-deeplabv3plus) |

 Zero-shot image segmentation model:

| Network  | Size (MB) | Example Location |
| ------------- | ------------- |------------- |
| [segmentAnythingModel](https://www.mathworks.com/help/images/ref/segmentanythingmodel.html)  | 358 | [Doc](https://www.mathworks.com/help/images/getting-started-with-segment-anything-model.html) |


**Application Specific Semantic Segmentation Models**
| Network  | Application |Size (MB) |Location |Example Output |
| ------------- | ------------- | ------------- |------------- |------------- |
| [U-net](https://www.mathworks.com/help/images/develop-raw-camera-processing-pipeline-using-deep-learning.html)   | Raw Camera Processing |31 |[Doc](https://www.mathworks.com/help/images/develop-raw-camera-processing-pipeline-using-deep-learning.html) | <img src="Images/rawimage.png" width=150>|
| [3-D U-net](https://www.mathworks.com/help/images/segment-3d-brain-tumor-using-deep-learning.html) | Brain Tumor Segmentation  | 56.2 | [Doc](https://www.mathworks.com/help/images/segment-3d-brain-tumor-using-deep-learning.html) | <img src="Images/Segment3DBrainTumor.gif" width=150>|
| [AdaptSeg (GAN)](https://www.mathworks.com/help/deeplearning/ug/train-deep-learning-semantic-segmentation-network-using-3d-simulation-data.html)  | Model tuning using 3-D simulation data | 54.4 | [Doc](https://www.mathworks.com/help/deeplearning/ug/train-deep-learning-semantic-segmentation-network-using-3d-simulation-data.html) |<img src="Images/adaptSeg.png" width=150>|


[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Instance Segmentation <a name="InstanceSegmentation"/>

Instance segmentation is an enhanced type of object detection that generates a segmentation map for each detected instance of an object. Instance segmentation treats individual objects as distinct entities, regardless of the class of the objects. In contrast, semantic segmentation considers all objects of the same class as belonging to a single entity.

Inputs are RGB images, outputs are pixel classifications (semantic maps), bounding boxes and classification labels.

![](Images/maskrcnn.png)

| Network | Object Classes | Location |
| ------------- | ------------- |------------- |
| [Mask R-CNN](https://www.mathworks.com/help/vision/ref/maskrcnn.html) | 80 | [Doc](https://www.mathworks.com/help/vision/ref/maskrcnn.html) <br />  [Github](https://github.com/matlab-deep-learning/mask-rcnn)|

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Image Translation <a name="ImageTranslation"/>

Image translation is the task of transferring styles and characteristics from one image domain to another. This technique can be extended to other image-to-image learning operations, such as image enhancement, image colorization, defect generation, and medical image analysis.

Inputs are images, outputs are translated RGB images. This example workflow shows how a semantic segmentation map input translates to a synthetic image via a pretrained model (Pix2PixHD):

![](Images/generativeimage.png)

| Network  | Application |Size (MB) |Location |Example Output |
| ------------- | ------------- | ------------- |------------- |------------- |
| [Pix2PixHD(CGAN)](https://www.mathworks.com/help/deeplearning/ug/generate-image-from-segmentation-map-using-deep-learning.html) | Synthetic Image Translation | 648 | [Doc](https://www.mathworks.com/help/deeplearning/ug/generate-image-from-segmentation-map-using-deep-learning.html) |<img src="Images/SynthesizeSegmentation.png" width=150> |
| [UNIT (GAN)](https://www.mathworks.com/help/images/unsupervised-day-to-dusk-image-translation-using-unit.html) | Day-to-Dusk Dusk-to-Day Image Translation | 72.5 | [Doc](https://www.mathworks.com/help/images/unsupervised-day-to-dusk-image-translation-using-unit.html) |<img src="Images/day2dusk.png" width=150>|
| [UNIT (GAN)](https://www.mathworks.com/help/images/unsupervised-medical-image-denoising-using-unit.html) | Medical Image Denoising | 72.4 | [Doc](https://www.mathworks.com/help/images/unsupervised-medical-image-denoising-using-unit.html) |<img src="Images/unit_imagedenoising.png" width=150>|
| [CycleGAN](https://www.mathworks.com/help/images/unsupervised-medical-image-denoising-using-cyclegan.html) | Medical Image Denoising | 75.3 | [Doc](https://www.mathworks.com/help/images/unsupervised-medical-image-denoising-using-cyclegan.html) |<img src="Images/cyclegan_imagedenoising.png" width=150>|
| [VDSR](https://www.mathworks.com/help/images/single-image-super-resolution-using-deep-learning.html) | Super Resolution (estimate a high-resolution image from a low-resolution image) | 2.4 | [Doc](https://www.mathworks.com/help/images/single-image-super-resolution-using-deep-learning.html) |<img src="Images/SuperResolution.png" width=150>|

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Pose Estimation <a name="PoseEstimation"/>

Pose estimation is a computer vision technique for localizing the position and orientation of an object using a fixed set of keypoints. 

All inputs are RGB images, outputs are heatmaps and part affinity fields (PAFs) which via post processing perform pose estimation.

![](Images/pose_workflow.png)

| Network  | Backbone Networks | Size (MB)| Location|
| ------------- | -------------| ------------- |------------- |
| [OpenPose](https://www.mathworks.com/help/deeplearning/ug/estimate-body-pose-using-deep-learning.html) | vgg19  | 14 | [Doc](https://www.mathworks.com/help/deeplearning/ug/estimate-body-pose-using-deep-learning.html) |
| [HR Net](https://www.mathworks.com/help/vision/ref/hrnetobjectkeypointdetector.html) | human-full-body-w32<br />human-full-body-w48  | 106.9<br />237.7 | [Doc](https://www.mathworks.com/help/vision/ref/hrnetobjectkeypointdetector.html) |


[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)
## 3D Reconstruction <a name="3DReconstruction"/>

3D reconstruction is the process of capturing the shape and appearance of real objects.

| Network  | Size (MB)| Location|Example Output|
| ------------- | ------------- |------------- |------------- |
| [NeRF](https://github.com/matlab-deep-learning/nerf)   | 3.78 |[GitHub](https://github.com/matlab-deep-learning/nerf)|![NeRF](Images/nerf.jpg) |

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Video Classification <a name="VideoClassification"/>

Video classification is a computer vision technique for classifying the action or content in a sequence of video frames. 

All inputs are Videos only or Video with Optical Flow data, outputs are gesture classifications and scores.

![](Images/video_workflow.png)

| Network  | Inputs | Size(MB) | Classifications (Human Actions)| Description | Location|
| ------------- | ------------- |------------- |------------- |------------- |------------- |
| [SlowFast](https://www.mathworks.com/help/vision/ref/slowfastvideoclassifier.html) | Video | 124 |400 |Faster convergence than Inflated-3D |[Doc](https://www.mathworks.com/help/vision/ref/slowfastvideoclassifier.html)
| [R(2+1)D](https://www.mathworks.com/help/vision/ref/r2plus1dvideoclassifier.html) | Video | 112 |400 |Faster convergence than Inflated-3D|[Doc](https://www.mathworks.com/help/vision/ref/r2plus1dvideoclassifier.html)
| [Inflated-3D](https://www.mathworks.com/help/vision/ref/inflated3dvideoclassifier.html) | Video & Optical Flow data | 91 | 400 |Accuracy of the classifier improves when combining optical flow and RGB data.| [Doc](https://www.mathworks.com/help/vision/ref/inflated3dvideoclassifier.html)

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Text Detection and Recognition <a name="textdetection"/>

Text detection is a computer vision technique used for locating instances of text within in images.

Inputs are RGB images, outputs are bounding boxes that identify regions of text.

![](Images/textdetect_workflow.png)

| Network  | Application | Size (MB) | Location |
| ------------- | ------------- |------------- |------------- |
| [CRAFT](https://www.mathworks.com/help/vision/ref/detecttextcraft.html)   | Trained to detect English, Korean, Italian, French, Arabic, German and Bangla (Indian).| 3.8 |[Doc](https://www.mathworks.com/help/vision/ref/detecttextcraft.html) <br /> [GitHub](https://github.com/matlab-deep-learning/Text-Detection-using-Deep-Learning) |

**Application Specific Text Detectors**

| Network  | Application | Size (MB) | Location |Example Output |
| ------------- | ------------- |------------- |------------- |------------- |
| [Seven Segment Digit Recognition](https://github.com/matlab-deep-learning/Seven-Segment-Digit-Recognition)   |Seven segment digit recognition using deep learning and OCR. This is helpful in industrial automation applications where digital displays are often surrounded with complex background. | 3.8 |[Doc](https://www.mathworks.com/help/vision/ref/ocr.html) <br /> [GitHub](https://github.com/matlab-deep-learning/Seven-Segment-Digit-Recognition)  |![](Images/7segment.png)|

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Transformers (Text) <a name="transformers"/>

Transformer pretained models have already learned to extract powerful and informative features features from text. Use them as a starting point to learn a new task using transfer learning. 

Inputs are sequences of text, outputs are text feature embeddings.

![](Images/transformer_workflow.png)


| Network  | Applications | Size (MB) | Location |
| ------------- | ------------- |------------ |------------ |
| [BERT](https://www.mathworks.com/help/textanalytics/ref/bert.html)   | Feature Extraction (Sentence and Word embedding), Text Classification, Token Classification, Masked Language Modeling, Question Answering |390 |[GitHub](https://github.com/matlab-deep-learning/transformer-models#bert-and-finbert) <br /> [Doc](https://www.mathworks.com/help/textanalytics/ref/bert.html) | 
| [all-MiniLM-L6-v2](https://www.mathworks.com/matlabcentral/fileexchange/156399-text-analytics-toolbox-model-for-all-minilm-l6-v2-network)   | Document Embedding, Clustering, Information Retrieval  |80 |[Doc](https://www.mathworks.com/matlabcentral/fileexchange/156399-text-analytics-toolbox-model-for-all-minilm-l6-v2-network)  | 
| [all-MiniLM-L12-v2](https://www.mathworks.com/matlabcentral/fileexchange/156394-text-analytics-toolbox-model-for-all-minilm-l12-v2-network)   | Document Embedding, Clustering, Information Retrieval  |120 |[Doc](https://www.mathworks.com/matlabcentral/fileexchange/156394-text-analytics-toolbox-model-for-all-minilm-l12-v2-network)  | 

**Application Specific Transformers**

| Network  | Application | Size (MB) | Location | Output Example |
| ------------- | ------------- | ------------- |------------- |------------- |
| [FinBERT](https://github.com/matlab-deep-learning/transformer-models#bert-and-finbert)   | The FinBERT model is a BERT model for financial sentiment analysis | 388 |[GitHub](https://github.com/matlab-deep-learning/transformer-models#bert-and-finbert) |![](Images/finbert.png)|
| [GPT-2](https://github.com/matlab-deep-learning/transformer-models#gpt-2) | The GPT-2 model is a decoder model used for text summarization.| 1.2GB |[GitHub](https://github.com/matlab-deep-learning/transformer-models#gpt-2) |![](Images/gpt2.png)|

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Audio Embeddings <a name="AudioEmbeddings"/>

Audio embedding pretrained models have already learned to extract powerful and informative features from audio signals. Use them as a starting point to learn a new task using transfer learning. 

Inputs are audio signals, outputs are audio feature embeddings.

![](Images/audio_workflow.png)

**Note 2: Since R2024a, please use the [audioPretrainedNetwork](https://www.mathworks.com/help/audio/ref/audiopretrainednetwork.html) function instead and specify the pretrained model**.
For example, use the following code to access VGGish:
```
net = audioPretrainedNetwork("vggish");
```

| Network  | Application | Size (MB) | Location |
| ------------- | ------------- | ------------- |------------- |
| [VGGish](https://www.mathworks.com/help/audio/ref/vggish.html)<sup>2<sup>   | Feature Embeddings | 257 |[Doc](https://www.mathworks.com/help/audio/ref/vggish.html) |
| [OpenL3](https://www.mathworks.com/help/audio/ref/openl3.html)<sup>2<sup>   | Feature Embeddings | 200 |[Doc](https://www.mathworks.com/help/audio/ref/openl3.html) |

## Application Specific Audio Models<a name="Application Specific Audio Models"/>

| Network  | Application | Size (MB) | Output Classes | Location | Output Example |
| ------------- | ------------- | ------------- |------------- |------------- |------------- |
| <a name="SoundClassification"/>[vadnet](https://www.mathworks.com/help/audio/ref/vadnet.html)<sup>2<sup>   | Voice Activity Detection (regression) | 0.427 | - |[Doc](https://www.mathworks.com/help/audio/ref/vadnet.html) |<img src="Images/vadnet.png" width=150>|
| <a name="SoundClassification"/>[YAMNet](https://www.mathworks.com/help/audio/ref/yamnet.html)<sup>2<sup>   | Sound Classification | 13.5 | 521 |[Doc](https://www.mathworks.com/help/audio/ref/yamnet.html) |<img src="Images/audio_classification.png" width=150>|
| <a name="PitchEstimation"/>[CREPE](https://www.mathworks.com/help/audio/ref/crepe.html)<sup>2<sup>   | Pitch Estimation (regression) | 132| - |[Doc](https://www.mathworks.com/help/audio/ref/crepe.html) |<img src="Images/pitch_estimation.png" width=150>|

## Speech to Text <a name="Speech2Text"/>

Speech-to-text models provide a fast, efficient method to convert spoken language into written text, enhancing accessibility for individuals with disabilities, enabling downstream tasks like text summarization and sentiment analysis, and streamlining documentation processes. As a key element of human-machine interfaces, including personal assistants, it allows for natural and intuitive interactions, enabling machines to understand and execute spoken commands, improving usability and broadening inclusivity across various applications.

Inputs are audio signals, outputs is text.

![](Images/wav2vec.png)

| Network  | Application | Size (MB) | Word Error Rate (WER) | Location | 
| ------------- | ------------- | ------------- |------------- |------------- |
| [wav2vec](https://github.com/matlab-deep-learning/wav2vec-2.0)   | Speech to Text | 236| 3.2 |[GitHub](https://github.com/matlab-deep-learning/wav2vec-2.0) |
| [deepspeech](https://github.com/matlab-deep-learning/deepspeech)   | Speech to Text | 167| 5.97 |[GitHub](https://github.com/matlab-deep-learning/deepspeech) |

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Lidar <a name="PointCloud"/>

Point cloud data is acquired by a variety of sensors, such as lidar, radar, and depth cameras. Training robust classifiers with point cloud data is challenging because of the sparsity of data per object, object occlusions, and sensor noise. Deep learning techniques have been shown to address many of these challenges by learning robust feature representations directly from point cloud data. 

Inputs are Lidar Point Clouds converted to five-channels, outputs are segmentation, classification or object detection results overlayed on point clouds.

![](Images/lidar_workflow.png)

| Network  | Application | Size (MB) | Object Classes | Location | 
| ------------- | ------------- | ------------- |------------- |------------- |
| [PointNet](https://www.mathworks.com/help/vision/ug/point-cloud-classification-using-pointnet-deep-learning.html)  | Classification | 5| 14 |[Doc](https://www.mathworks.com/help/vision/ug/point-cloud-classification-using-pointnet-deep-learning.html)|
| <a name="PointCloudSeg"/>[PointNet++](https://www.mathworks.com/help/lidar/ug/aerial-lidar-segmentation-using-pointnet-network.html)  | Segmentation | 3| 8 |[Doc](https://www.mathworks.com/help/lidar/ug/aerial-lidar-segmentation-using-pointnet-network.html)|
| [PointSeg](https://www.mathworks.com/help/deeplearning/ug/lidar-semantic-segmentation-using-pointseg.html)   | Segmentation | 14| 3 |[Doc](https://www.mathworks.com/help/vision/ug/point-cloud-classification-using-pointnet-deep-learning.html)|
| [SqueezeSegV2](https://www.mathworks.com/help/deeplearning/ug/lidar-semantic-segmentation-using-squeezesegv2.html)   | Segmentation |5| 12 |[Doc](https://www.mathworks.com/help/deeplearning/ug/lidar-semantic-segmentation-using-squeezesegv2.html) |
| [SalsaNext](https://github.com/matlab-deep-learning/pretrained-salsanext)   | Segmentation |20.9 | 13 |[GitHub](https://github.com/matlab-deep-learning/pretrained-salsanext)|
| <a name="PointCloudObj"/>[PointPillars](https://www.mathworks.com/help/lidar/ug/object-detection-using-pointpillars-network.html)   | Object Detection | 8| 3 |[Doc](https://www.mathworks.com/help/lidar/ug/object-detection-using-pointpillars-network.html)|
| [Complex YOLO v4](https://github.com/matlab-deep-learning/Lidar-object-detection-using-complex-yolov4)   | Object Detection | 233 (complex-yolov4) <br /> 21 (tiny-complex-yolov4) | 3 |[GitHub](https://github.com/matlab-deep-learning/Lidar-object-detection-using-complex-yolov4)|

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Manipulator Motion Planning <a name="ManipMotionPlanning"/>

Manipulator motion planning is a technique used to plan a trajectory for a robotic arm from a start position to a goal position in an obstacle environment.
  
Pretrained deep learning models have learned to plan such trajectories  for repetitive tasks such as picking and placing of objects, leading to speed ups over traditional algorithms.

Inputs are start configuration, goal configuration and obstacle environment encoding for the robot, outputs are intermediate trajectory guesses.

![](Images/motion_planning_workflow.svg)

| Network  | Application | Size (MB)| Location|
| --- | --- | --- | --- |
| [Deep-Learning-Based CHOMP (DLCHOMP)](https://www.mathworks.com/help/releases/R2024a/robotics/ref/dlchomp.html)   | Trajectory Prediction | 25 | [Doc](https://www.mathworks.com/help/releases/R2024a/robotics/ref/dlchomp.html)<br />[GitHub](https://github.com/matlab-deep-learning/pretrained-dlchomp) |



[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)


## Path Planning with Motion Planning Networks <a name="PathPlanningMPNet"/>

Motion Planning Networks (MPNet) is a deep-learning-based approach for finding optimal paths between a start point and goal point in motion planning problems. MPNet is a deep neural network that can be trained on multiple environments to learn optimal paths between various states in the environments. The MPNet uses this prior knowledge to,
- Generate informed samples between two states in an unknown test environment. These samples can be used with sampling-based motion planners such as optimal rapidly-exploring random trees (RRT*) for path planning.
- Compute collision-free path between two states in an unknown test environment. MPNet based path planner is more efficient than the classical path planners such as the RRT*.

To know more please visit [Get Started with Motion Planning Networks](https://in.mathworks.com/help/nav/ug/get-started-with-motion-planning-networks.html)

![](Images/mpnetarchitecture.png)

| Network                                                                                                     | Application   | Size (MB) | Location                                                                                                |
| ----------------------------------------------------------------------------------------------------------- | ------------- | --------- | ------------------------------------------------------------------------------------------------------- |
| [mazeMapTrainedMPNET](https://www.mathworks.com/help/nav/ug/get-started-with-motion-planning-networks.html) | Path Planning | 0.23      | [Doc](https://www.mathworks.com/help/nav/ug/train-deep-learning-based-sampler-for-motion-planning.html) |

[Back to top](https://github.com/matlab-deep-learning/MATLAB-Deep-Learning-Model-Hub#matlab-deep-learning-model-hub)

## Model requests
If you'd like to request MATLAB support for additional pretrained models, [**please create an issue from this repo**](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue). 

Alternatively send the request through to:

[Jianghao Wang](https://www.linkedin.com/in/jianghao-wang-896aa1a4/) <br /> 
Deep Learning Product Manager <br /> 
jianghaw@mathworks.com

Copyright 2024, The MathWorks, Inc.

