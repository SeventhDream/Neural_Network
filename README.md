# Building a Neural Network from Scratch

# Objective:
- Build a 10-class classifier to identify MNIST images of handwritten digits ranign from 0 to 9.

# Trianing Data Structure
- Each image is 28x28 resolution (784 pixels).
- Data consists of image pixel values ranging from 0 to 255.
- Data matrix transposed to align each images' pixel data to a separate column.

## Neural Network Architecture
```
m = total #no. of images
```
- m nodes on input layer.
- 2 hidden layers each with 10 nodes initialised with random weights and biases.
- m nodes on output layer.

## Forward Propagation
- Inputs image(s) and outputs classification prediciton(s).
- W = Node weight.
- B = Node bias.
- A = Previous node output.

$y = mx + c = WA + B$

$A_0 = X$ (Input layer input is the original image data).

$Z_1 = W_1A_0 + B_1$

$A_1 = \sigma(Z_1) = ReLU(Z_1)$

$Z_2 = W_2A_1 + B_2$
$A_2 = softmax(Z_2)$ (Final output is one-hot encoded).

## Backwards Propagation
- Optimise the weights and biases.
- $\frac{1}{m}$ used for multiple images.

$Z_2 = A_2 - Y$

$-> dW_2 = \frac{1}{m}dZ_2A_1^T = \frac{1}{m}(A_2 - Y)A_1^T$

$-> dB_2 = \frac{1}{m}\sum dZ_2 = \frac{1}{m}\sum(A_2 - Y)$

$dZ_1 = W_2^TdZ_2 * ReLU(Z_1)$

$-> W_1 = \frac{1}{m}dZ_1A_0^T = \frac{1}{m}W_2^TdZ_2 * ReLU^{'}(Z_1)A_0^T = \frac{1}{m}W_2^T(A_2 - Y) * ReLU^{'}(Z_1)A_0^T$

$-> dB_1 = \frac{1}{m}\sum dZ_1 = \frac{1}{m}\sum W_2^T(A_2 - Y) * ReLU^{'}(Z_1)$

## Plot Output
- Neural network model achieves a classification accuracy of 89.9% during training for 2000 iterations and 88.9% accuracy when tested aghainst validation dataset.
  - Further tuning required to further increase accuracy to desirable threshold (>99.5%). 
![Training Accuracy Plot](https://github.com/SeventhDream/Neural_Network/blob/main/training_plot.png?raw=true)


