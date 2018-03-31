## Caffe Computation Graph Optimization. (BETA)
### Intro :
**To be clear :** If you are willing to know something about computation graph optimization, [NNVM](http://nnvm.tvmlang.org/) or [XLA](https://www.tensorflow.org/performance/xla/) should be a good start. Here, we just focus on how to squeeze the redundant operations out of topological order computation graph, which is prototxt in Caffe. **Just Tricks, No Theory.**

### How to optimize Caffe computation graph ?
- Operation Fusion **`*`**:
  - conv/fc + bn + scale => conv/fc   
  ```a(((Wx+c)-mu)/std)+b = [aW/std]x+[a(c-mu)/std+b] = W'x+c'```
  - conv/fc + bn => conv/fc   
  ```((Wx+c)-mu)/std = [W/std]x+[(c-mu)/std] = W'x+c'```
  - data + conv/fc => conv/fc **`**`**   
  ```W(k(x-mean))+c = [kW]x+[-kW*mean+c] = W'x+c'```
- In-place Computation:
  - Activation : ReLU ; TanH ; SoftMax ; PReLU
  - Affine : Batch-Normalization ; Scale
- Layer Elimination:
  - Dropout

**`*`**: We won't discuss layer reconstruction, such as conv-bn-relu layer. In this repository, one may accelerate models without writing a single line of code.   
**`**`**: First convolution layer must hold **`pad==0`**, or the fusion of mean value subtraction may lead to performance loss. But scale factor fusion, is free to be utlized without those constraints. Taking 2-D input matrix as an example, 
```matlab
% Matlab 
>> mean = 122;
>> X = mean*ones(3);
>> W = rand(2);
>> conv2(W,X,'full') % padding != 0 leading to different output
ans =
  116.8158  136.0446  136.0446   19.2288
  234.5322  372.1734  372.1734  137.6411
  234.5322  372.1734  372.1734  137.6411
  117.7164  236.1287  236.1287  118.4123
>> conv2(W,X,'same') % new_bias = 372.1734 with padding == 0
ans =
  372.1734  372.1734
  372.1734  372.1734
```

### Visualisation :
We use a cifar-10 network to go into the details (batchsize=10 with forward and backward). 

Optimization | Netscope | CPU forward (ms) | GPU forward (ms) | GPU Memory (MB)
------------ | ------------- | ------------- | -------------- | ---------
Naive | [view](http://ethereon.github.io/netscope/#/gist/46e9a5a337b67f4e7cfcd1b04137a8a9) | 26.20 | 2.766 | 168
In-place | [view](http://ethereon.github.io/netscope/#/gist/c964c5e940ac21c5a8cc67257b345d7f) | 26.29 | 2.734 | 165
Fusion | [view](http://ethereon.github.io/netscope/#/gist/409198ed27b1f26595f329aa5e550016) | **23.16** | **2.070** | **160**

*CPU : Intel® Xeon(R) CPU E3-1220 v5 @ 3.00GHz × 4 + OpenBLAS*   
*GPU : K80 + CUDA8.0 + cuDNN*   

**What is the benefit of in-place optimization ?**   
In fact, it is a problem inherent in Caffe. As annotation said:   
> In-place computation; need to store bottom data before overwriting it. Note that this is only necessary for Backward; we could skip this if not doing Backward, but Caffe currently provides no way of knowing whether we'll need to do Backward at the time of the Forward call.

There are some redundant backup memorys at inference time. To further reduce the memory footprint, we need to define a novel memory allocation mechanism, which will be discussed in the future.   
In this case, we use in-place optimization to simplify data dependecy, then we can merge BN and Scale easily.   

**Workhorse Networks :**   

GPU forward (ms) | [AlexNet-BN](https://github.com/HolmesShuan/AlexNet-BN-Caffemodel-on-ImageNet) | [ResNet-18](https://github.com/HolmesShuan/ResNet-18-Caffemodel-on-ImageNet) | [MobileNet-v1](https://github.com/shicai/MobileNet-Caffe) 
------- | --------------- | ------------------ | --------------------
Naive | 6.981 | 13.769 | 14.550
OPT   | 5.347 | 8.865 | 7.167
Acceleration | **1.31x** | **1.55x** | **2.03x**

*mean time of 1K iterations with batchsize=1*
### How to use ?
```shell
pip install configparser --user
pip install numpy --user
# install Caffe
cd /your/caffe/path
make pycaffe
# edit config.ini 
python main.py
# check your target model and prototxt path
# test accuracy, which should be the same as original model 
```
Here is an example of `config.ini`:
```ini
[DEFAULT]
pycaffe_path = /home/shuan/caffe-master/python
# MODEL PATH
# original model and prototxt path
original_prototxt_path = ./model/train_val.prototxt
original_model_path = ./model/cifar10.caffemodel
# target model and prototxt path
optimized_prototxt_path = ./model/opt_train_val.prototxt
optimized_model_path = ./model/opt_cifar10.caffemodel

# MERGE BN SCALE 
merge_bn_scale = yes 
# yes / no

# MERGE INPUT PREPROCESS
merge_input_scale = yes
merge_input_mean = yes
# yes / no
# using mean() to calc mean vector [R G B] 
mean_proto_path = ./data/mean.npy 
# if mean.npy didn't exist, we will use given mean values below
mean_R = 123.0
mean_G = 117.0
mean_B = 104.0
# INPUT SCALE FACTOR, e.g. MobileNet used 0.017 to scale raw data.
input_scale = 0.017
# INPUT IMAGE SIZE
H = 224
W = 224
# setting H and W explicitly, in case your prototxt was not the format of deploy.prototxt
C = 3
# 3 for RGB; 1 for Gray

# ELIMINATE DROPOUT
merge_dropout = yes
# yes / no

# MERGE IN-PLACE MEMORY
merge_inplace_memory = yes
# yes / no
# inplace operation memory reuse

```
