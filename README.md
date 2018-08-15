# DIRNet-tensorflow
Tensorflow implementation of DIRNet
- <b>[CAUTION] This implementation is actually different from the original paper. This implementation uses a simple bicubic interpolation not cubic spline interpolation. They can be different in details. If you want to more precise experiments, you need to change the codes. </b>

![alt tag](misc/DIRNet.png)

## Usage
```
# Training
python train.py
```
Intermediate results and model checkpoint can be found in ```tmp``` and ```ckpt``` each.

```
# Evaluation
python deploy.py 
```
Evaluation results can be found in ```result```.

## References
- [End-to-End Unsupervised Deformable Image Registration with a Convolutional Neural Network](https://arxiv.org/abs/1704.06065)
- [Spatial Transformer Network](https://arxiv.org/abs/1704.06065)
- [Tensorflow implementation of STN](https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py)
- [Tensorflow implementation of bicubic interpolation](https://github.com/iwyoo/bicubic_interp-tensorflow)

## Author
Inwan Yoo / iwyoo@unist.ac.kr
