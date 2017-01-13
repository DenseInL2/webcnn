
# WebCNN

WebCNN is a browser-based Convolutional Neural Network framework. This is a personal project in the earliest
stages of development, which I'm sharing publicly for those with academic interest. Here is what is currently implemented and tested:

- Input image layer of arbitary dimensions, which accepts RGB ImageData for input. Full RGB is supported, as well as using the red channel for greyscale images
- Convolutional Layers with square kernels of odd dimension ( 3x3, 5x5, etc. ), stride of 1, no zero padding, and ReLU activation
- Max pooling layers with variable size and stride
- Fully-connected layers with tanh or linear activation on any layer, and additionally softmax classifier support on the final layer
- A numerical input layer is implemented for debugging and testing small non-convolutional neural networks
- Support for traditional SGD with momentum and L2 regularization
- JavaScript/ES6 implementation
- Batch processing of input (an image input layer takes arrays of ImageData and classification values)
- Example webpage code for training on the MNIST digit dataset, and recognizing digits

Here is what I have planned for future implementation, in rough priority order:
- Zero-padding and adjustable stride values for convolutional layers
- Phase 1 of WebGL support, just for convolution operations
- Phase 2 of WebGL support for the entire network to train on the GPU
- Example webpages for CIFAR and ImageNet datasets, and possibly others
- Implementation of "dreaming" and "style transfer" modes for backpropagating changes to input images
- Support for additional classifiers
- Support for more variants of optimized gradient descent, such as Nesterov momentum udpates, RMSProp, Adagrad, Adadelta, Adam, etc.
- Support for more flexible weight initialization options

**Note**: I have not done extensive testing across multiple browsers or operating systems. I have done nearly all development
of this project on a MacBook Pro Mid-2014 with 2.5GHz i7 and Nvidia 750M GPU in Chrome 55.0. I started the project using strict ES6 syntax,
with let and const declarations, but was forced to fall back to using var declarations in places where I encountered the newer style
preventing Chrome from optimizing the code.

I'm also a C++, C# and Java developer by day, which is why my code looks more like Java written in JavaScript than the work of
a native JavaScripter. I work daily with projects using the class inheritance OOP paradigm. It's on my todo list to learn about the
more functional ways of programming in JS, but for now I'm focusing on making something that works in the way I know. I have written,
debugged, and ported the earliest sections of this project from Java because I found it easier to debug in a more familiar IDE, before
I discovered WebStorm.
