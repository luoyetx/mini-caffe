Mini-Caffe-Examples
==================

Projects use Mini-Caffe as a C++ library to run CNNs.

### deeplandmark

Detect facial landmarks with Mini-Caffe. Caffe models are trained by [luoyetx/deep-landmark](https://github.com/luoyetx/deep-landmark). A video test can be viewed [here](https://youtu.be/oNiAtu0erEk).

### G in WGAN

Generate anime face using WGAN. Model is trained by [luoyetx/WGAN](https://github.com/luoyetx/WGAN) and converted to Caffe model.

### R-FCN

[Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409) is converted from [Orpine/py-R-FCN](https://github.com/Orpine/py-R-FCN). Donwload `resnet50_rfcn_final.caffemodel` then you can run the code.

### Build

You need [OpenCV](http://opencv.org/) to build examples.

```
$ mkdir build
$ cd build
$ cmake ..
```
