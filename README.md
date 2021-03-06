# Neural Transfer

This project implements the neural transfer algorithm proposed by Gatys
et al. in [1]. Gradient descent is performed on a noise image to
match the features of a content image and a style image produced by an encoder network. 

- The project is written in Python ```3.7``` and uses PyTorch ```1.1``` 
(also working with PyTorch ```1.3```).

- ````requirements.txt```` lists the python packages needed to run the 
project. 

### Network

The network architecture resembles the proposed one of Gatys et al. A VGG-19
network is utilized and produces the feature responses of a noise image, a 
content image and a style image. 

### Loss
 
Two losses ensure the stylization of the content image in the style of the style
image and are both applied to the noise image. A content loss measures the MSE 
between noise features and content features. A style loss measures the MSE between
the Gram matrix of noise features and the Gram matrix of style features summed up
over the specified layers in the network.

#### Moment Alignment
The idea of the project is to compare the Gram matrix loss with a so-called moment 
loss that measures the difference between the first ````N```` moments of noise
features and style features.  

A further explanation as well as analysis can be found in the pdf-file of this 
repository.

### Usage

The ``configurations``-folder specifies two configurations that can be used to 
produce transfer image. The project only gets the exact path to the 
configuration that is used e.g. ```python main.py './configurations/train.yaml'```.

- ``train.yaml`` produces transfer images from content and style pairs.

- ```train_mmd.yaml``` produces transfer images from content and style pairs. The MMD 
is utilized as loss function. Please also see [2].

### Additional Information

This project is part of a bachelor thesis which was submitted in August 2019. This 
neural trainsfer network makes up one chapter of the final thesis. A slightly modified 
version of the chapter can be found in this repository as a pdf-file. Also, the chapter 
introduces all related formulas to this work. 

The final thesis can be found [here](https://jzenn.github.io/projects/bsc-thesis) in a corrected and modified version.

### References

[1] L. Gatys, A. Ecker, and M. Bethge. Texture synthesis using convolutional neural networks. In Conference on Neural Information Processing Systems (NIPS), 2015.

[2] A. Gretton, K. Borgwardt, M. Rasch, B. Schölkopf, and A. Smola. A kernel two-sample test. J. Mach. Learn. Res., 2012. 