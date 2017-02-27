Mini-Caffe
==========

Python Package for Mini-Caffe.

### Install

Numpy is needed for mini-caffe. Before we can install package, please compile Mini-Caffe first.

```
$ python setup.py install
```

### Usage

```python
import numpy as np
import minicaffe as mcaffe
# create network
net = mcaffe.Net('/path/to/xxx.prototxt', '/path/to/xxx.caffemodel')
# get your network input blob
data_blob = net.get_blob('your_input_blob')
# alloc memory
data_blob.reshape(1, 3, 224, 224)
# fill the data, data type should be float32
data_blob.data[...] = np.random.rand((1, 3, 224, 224)).astype(np.float32)
# forward network
net.forward()
# get output blob
output_blob = net.get_blob('your_output_blob')
# get numpy data
output = output_blob.data
```
