# MLnumpy

MLnumpy is a simple project aimed at understanding the fundamental mechanisms of neural networks from scratch.
It provides a pure implementation of neural networks using the NumPy library.

The project focuses on building neural networks from the ground up, allowing users to explore and gain insights into the inner workings of these powerful models.
By implementing neural networks using NumPy, MLnumpy offers a hands-on approach to learning the underlying principles and algorithms that drive deep learning.

Whether you're a beginner looking to grasp the basics of neural networks or an experienced practitioner seeking a deeper understanding,
MLnumpy provides a clear and concise implementation that can serve as a stepping stone for further exploration in the field of machine learning.


## Installation

To install the project, follow these steps:

1. Clone the repository.
2. Create a Python 3.10+ environment.
3. Install the required packages by running the following command: `pip install -r requirements.txt`.


## Documentation

The MLnumpy framework consists of various components that represent different aspects of neural networks.


### Layers

* **Dense** implements the dense layer of a neural network.
  It is also known as the fully connected layer.
  It connects every neuron in the previous layer to every neuron in the current layer.
  The output of this layer is calculated as follows:

  ![dense](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\bg_white&space;\Large&space;\text{output}=\text{activation}(\text{dot}(\text{input},&space;\text{weights})&space;+&space;\text{bias}))

  Here, `dot` represents the dot product between the input and weights, and `activation` is the activation function applied to the result.

* **Flatten** implements the flatten layer, which reshapes the input tensor into a 1-dimensional array.
  It is typically used to transition from convolutional layers to fully connected layers.
  The input tensor is flattened according to the following formula:

  ![Flatten](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\bg_white&space;\Large&space;\text{output}=\text{input.reshape}(\text{batch\\_size},-1))

  The `-1` parameter in `reshape` automatically calculates the size of the flattened dimension.

* **Convolution2D** implements the 2D convolutional layer of a neural network.
  Convolutional layers are commonly used in image processing tasks.
  The output of this layer is obtained through a convolution operation between the input tensor and a set of learnable filters.
  The formula for computing the output feature map is as follows:

  ![Convolution2D](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\bg_white&space;\Large&space;\text{output}=\text{activation}(\text{convolution}(\text{input},&space;\text{filters})&space;+&space;\text{bias}))

  The `convolution` operation involves sliding the filters over the input tensor, performing element-wise multiplications and summing the results.

* **Pooling2D** implements the 2D pooling layer, which performs downsampling on the input tensor.

  - Max pooling is a commonly used pooling technique where the output of the pooling layer is obtained by selecting the maximum value within each pooling region.
    The formula for max pooling is as follows:

    ![MaxPooling2D](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\bg_white&space;\Large&space;\text{output}=\text{max\\_pooling}(\text{input},&space;\text{pool\\_size}))

    In the `max_pooling` operation, the input tensor is divided into non-overlapping regions, and the maximum value within each region is selected as the output.

  - Additionally, there is average pooling, which calculates the average value within each pooling region.
    The formula for average pooling is as follows:

    ![AvgPooling2D](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\bg_white&space;\Large&space;\text{output}=\text{avg\\_pooling}(\text{input},&space;\text{pool\\_size}))

    In the `avg_pooling` operation, similar to max pooling, the input tensor is divided into non-overlapping regions,
    but this time the average value within each region is selected as the output.


### Initializers

Initializers are used to set the initial values of the weights and biases in neural network layers.

* **Normal** generates random values from a normal distribution with mean `0` and standard deviation `1`.
  The formula for generating a random weight or bias value is as follows:

  ![normal](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\bg_white&space;\Large&space;\text{value}=\text{random.normal}(0,1))

* **Xavier** implements the Xavier initialization method, which is designed to keep the variances of the inputs and outputs of each layer approximately the same.
  The formula for generating a random weight or bias value using Xavier initialization is as follows:

  ![xavier](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\bg_white&space;\Large&space;\text{value}=\text{random.normal}\left(0,\sqrt{\frac{2}{\text{fan\\_in}+\text{fan\\_out}}}\right))

  Here, `fan_in` is the number of input units in the weight tensor, and `fan_out` is the number of output units.

* **Zero** sets all weights and biases to `0`.
  The formula for initializing weights and biases to zero is as follows:

  ![zero](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\bg_white&space;\Large&space;\text{value}=0)


### Activations

Activation functions introduce non-linearity to the neural network, enabling it to learn complex patterns and make nonlinear predictions.

* **Identity** function simply returns the input as the output without any transformation.
  The formula for the identity activation function is as follows:

  ![identity](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{output}=\text{input})

* **Rectified Linear Unit (ReLU)** function returns the maximum of `0` and the input value.
  It is commonly used in deep learning models to introduce non-linearity.
  The formula for the ReLU activation function is as follows:

  ![relu](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{output}=\max(0,\text{input}))

* **Sigmoid** function maps the input to a value between `0` and `1` using the sigmoid function.
  It is often used in binary classification problems.
  The formula for the sigmoid activation function is as follows:

  ![sigmoid](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{output}=\frac{1}{1+\exp(-\text{input})})

* **Softmax** function is used in multi-class classification problems to convert raw predictions into probabilities.
  It exponentiates each input value and normalizes them to sum up to `1`.
  The formula for the softmax activation function is as follows:

  ![softmax](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{output}=\frac{\exp(\text{input})}{\sum_{i=1}^{n}\exp(\text{input}_i)})

* **Hyperbolic tangent (Tanh)** function maps the input to a value between `-1` and `1` using the hyperbolic tangent function.
  It is commonly used in deep learning models to introduce non-linearity.
  The formula for the tanh activation function is as follows:

  ![tanh](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{output}=\frac{\exp(\text{input})-\exp(-\text{input})}{\exp(\text{input})+\exp(-\text{input})})


### Models

The models module provides a high-level interface for building neural network models.

* **Sequential Model** allows you to stack layers sequentially.
  It provides an easy way to build neural networks by adding layers one by one.
  Each layer in the Sequential model takes the output of the previous layer as its input.


### Losses

Loss functions play a crucial role in quantifying the disparity between predicted and actual values in a neural network.
They help in training the network by providing an objective measure of how well it is performing.

* **Mean Squared Error (MSE)** is a commonly used loss function for regression problems.
  It calculates the average squared difference between the predicted and actual values.
  The formula for MSE loss is expressed as:

  ![MSE](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{loss}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y_i})^2)

  Here, ![N](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;N) represents the number of samples in the dataset,
  ![y_i](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;y_i) denotes the actual value,
  and ![`\hat{y_i}`](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\hat{y_i}) signifies the predicted value for
  the ![`i`-th sample](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;i\text{-th%20sample}).

* **Cross-Entropy (CE)** is employed in multi-class classification problems.
  It measures the dissimilarity between the predicted probability distribution and the actual distribution.
  The formula for CE loss is given by:

  ![CE](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{loss}=-\sum_{i=1}^{C}y_i\log(\hat{y_i}))

  In the equation above, ![C](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;C) represents the number of classes,
  ![y_i](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;y_i) corresponds to the actual probability of the ![i](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;i)-th class,
  and ![`\hat{y_i}`](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\hat{y_i}) signifies the predicted probability for
  the ![i](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;i)-th class.

* **Softmax Cross-Entropy (SCE)** is another variant of the cross-entropy loss function often used in multi-class classification tasks.
  It combines the softmax activation function and cross-entropy loss to provide a more robust and numerical stable loss measure.
  The formula for softmax cross-entropy loss can be written as:

  ![SoftmaxCE](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{loss}=-\sum_{i=1}^{C}y_i\log\left(\frac{e^{\hat{y_i}}}{\sum_{j=1}^{C}e^{\hat{y_j}}}\right))

  Here, ![C](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;C) denotes the number of classes,
  ![y_i](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;y_i) represents the actual probability of
  the ![i](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;i)-th class,
  and ![`\hat{y_i}`](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\hat{y_i}) signifies the predicted score for
  the ![i](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;i)-th class before applying the softmax activation function.
  The softmax function ensures that the predicted scores form a valid probability distribution by exponentiating them and normalizing the sum of all scores.

These loss functions are essential tools in training neural networks and optimizing their performance.
By minimizing the chosen loss function, the network learns to make more accurate predictions and generalize better to unseen data.


### Optimizers

* **Stochastic Gradient Descent (SGD)** optimizer updates the weights in the opposite direction proportional to the gradient of the loss multiplied by
  the learning rate ![\eta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta) (a constant hyperparameter, default equals `0.1`).

  ![SGD](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\theta=\theta-\eta\nabla_\theta{J(\theta)}),

  where ![theta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\theta) represents the learnable weights,
  ![eta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta) is the learning rate,
  and ![nabla_theta_J](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\nabla_\theta{J(\theta)}) is the gradient of the loss function.

* **SGD with Momentum** takes into account the weight change from the previous step by introducing an additional variable called velocity and
  a constant hyperparameter called the momentum coefficient. It helps overcome flat regions (plateaus) in the function landscape to find the optimal minimum.
  The hyperparameters are set as follows: ![gamma](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\gamma=0.9), ![eta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta=0.01).

  ![SGD momentum](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;v_t=\gamma{v_{t-1}}+\eta\nabla_\theta{J(\theta)}),

  ![SGD momentum](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\theta=\theta-v_t),

  where ![v_t](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;v_t) is the velocity in the current step,
  ![v_{t-1}](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;v_{t-1}) is the velocity from the previous step,
  ![eta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta) is the learning rate,
  and ![nabla_theta_J](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\nabla_\theta{J(\theta)}) is the gradient of the loss function.

* **Nesterov Accelerated Gradient (NAG)** applies Nesterov's momentum, which provides stronger convergence for convex functions
  and performs slightly better than plain momentum in practice.
  It assumes "looking ahead" by calculating the gradient of the loss with respect to the future position of the parameter vector (rather than its current position),
  which prevents excessive acceleration and the risk of overshooting the minimum.
  The hyperparameters ![gamma](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\gamma) and ![eta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta) are set to the same values as plain momentum.

  ![NAG](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;v_t=\gamma{v_{t-1}}+\eta\nabla_\theta{J(\theta-\gamma{v_{t-1}})}),

  ![NAG](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\theta=\theta-v_t),

  In practice, Nesterov's momentum is often calculated using the transformed above formulas:

  ![NAG](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;v_t=\gamma{v_{t-1}}+\eta\nabla_\theta{J(\theta)}),

  ![NAG](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\theta=\theta-\gamma{v_{t-1}}+(1+\gamma)v_t),

  where ![eta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta) is the learning rate (default value of `0.001`),
  ![g_t](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;g_t) is the gradient in the current step,
  ![G_t](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;G_t) is the sum of squared gradients up to the current step,
  and ![eta_w](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta_w) is the variable learning rate.

* **Adagrad** adapts the learning rate depending on the history of weight changes.
  If weight changes have been infrequent, the learning rate will be larger, and for frequent changes, it will be smaller.

  ![Adagrad](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\theta_{t+1}=\theta_t-\frac{\eta_w}{\sqrt{G_t+\epsilon}}g_t),

  ![Adagrad](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta_w=\frac{\eta}{\sqrt{G_t+\epsilon}}g_t),

  ![Adagrad](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;G_t=\sum{(g_t)^2}),

  In this case, ![eta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta) represents the learning rate, with a default value of 0.001.
  ![g_t](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;g_t) is the gradient in the current step,
  ![G_t](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;G_t) is the sum of squared gradients up to the current step,
  and ![eta_w](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta_w) is the variable learning rate.

* **Adadelta** is an extension of Adagrad that prevents a large decrease in the learning rate.
  Instead of accumulating the history of all past gradients, it sums their moving average within a sliding window of past training iterations.
  In the original version, it does not require specifying a learning rate.
  The constants used in the formulas are as follows: ![gamma](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\gamma=0.95)
  and ![epsilon](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\epsilon=10^{-6}).

  ![Adadelta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\Delta{\theta_t}=-\frac{RMS[\delta{\theta}]_{t-1}}{RMS[g]_t}g_t),

  ![Adadelta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;RMS[g]_t=\gamma{E[g]^2_{t-1}}+(1-\gamma)g^2_t),

  ![Adadelta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;RMS[\Delta{\theta}]_t=\sqrt{E[\Delta{\theta^2}]_t+\epsilon}),

  where ![RMS_g_t](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;RMS[g]_t) (root mean square) is the gradient accumulator,
  ![g_t](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;g_t) is the gradient in the current step,
  ![Delta_theta_t](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\Delta{\theta_t}) is the weight update value,
  and ![RMS_Delta_theta_t](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;RMS[\Delta{\theta}]_t) is the accumulator of weight changes.

* **Adam** (Adaptive Moment Estimator) is one of the commonly used optimization methods in neural network training.
  The constants used in the formulas are as follows: ![eta](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\eta=0.001),
  ![beta_1](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\beta_1=0.9),
  ![beta_2](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\beta_2=0.999),
  ![epsilon](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\epsilon=10^{-8}).

  ![Adam](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;m_t=\beta_1m_{t-1}+(1-\beta_1)g_t),

  ![Adam](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;v_t=\beta_2v_{t-1}+(1-\beta_2)g^2_t),

  ![Adam](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\hat{m}_t=\frac{m_t}{1-\beta^t_1}),

  ![Adam](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\hat{v}_t=\frac{v_t}{1-\beta^t_2}),

  ![Adam](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t),

  where ![mt](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;m_t) and ![vt](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;v_t)
  are the first and second moments of the gradients, ![hat_mt](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\hat{m}_t)
  and ![hat_vt](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\hat{v}_t) are bias-corrected versions,
  ![gt](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;g_t) is the gradient in the current step,
  and ![theta_t](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\theta_t) is the weight in the current step.


### Metrics

The metrics module provides implementations of evaluation metrics for classification problems.

* **Accuracy** measures the proportion of correctly classified instances to the total number of instances. It is calculated as:

  ![Accuracy](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{accuracy}=\frac{\text{true\\_positives}+\text{true\\_negatives}}{\text{total\\_instances}})

* **Precision** measures the proportion of true positive predictions to the total number of positive predictions. It is calculated as:

  ![Precision](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{precision}=\frac{\text{true\\_positives}}{\text{true\\_positives}+\text{false\\_positives}})

* **Recall** measures the proportion of true positive predictions to the total number of actual positive instances. It is calculated as:

  ![Recall](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{recall}=\frac{\text{true\\_positives}}{\text{true\\_positives}+\text{false\\_negatives}})

* **F1 score** is the harmonic mean of precision and recall. It provides a balanced measure between precision and recall. It is calculated as:

  ![F1 Score](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\Large&space;\text{F1\\_score}=2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}})

These metrics are commonly used to evaluate the performance of classification models.
They provide insights into different aspects of the model's predictions, such as overall accuracy, precision in identifying positive instances,
recall in capturing all positive instances, and the balanced trade-off between precision and recall given by the F1 score.


### Callbacks

Callbacks are classes that are called at specific points during the training process.
They enrich the training procedure by adding additional functionality, such as displaying progress bars and saving the best-performing models.


### Datasets

The datasets module provides functionality for loading and preparing datasets for training and evaluation.

<img src="assets/mnist_dataset.png" alt="MNIST Dataset" align="right" width="350" />

* **MNIST** – dataset consists of a large collection of handwritten digits from 0 to 9.
It serves as a benchmark dataset for developing and evaluating various machine learning algorithms, particularly in the domain of image classification.
Each digit in the dataset is represented as a grayscale image of size 28x28 pixels.


### Utils

The utils module is a collection of utility functions that are designed to be general-purpose and not specific to any particular component.
It includes a variety of functions that can assist in common tasks such as data preprocessing, mathematical operations, visualization, and more.
These functions are designed to simplify the development process and enhance the overall usability of MLnumpy.


## Examples

In the MLnumpy framework, several example scripts are provided to demonstrate the usage and capabilities of the library.
These examples cover various aspects of neural networks and serve as a starting point for building and experimenting with your own models.
Here are two key examples:


### Training a Convolutional Neural Network (CNN)

To train the first model in the MLnumpy framework, you can run the following command in your terminal:

```bash
PYTHONPATH=. python examples/train_conv_net.py
```

This command executes the `train_conv_net.py` script, which trains a convolutional neural network.
The necessary data will be automatically downloaded to the `data` folder, and the training artifacts will be saved to the `experiments` directory.
By running this example, you can observe the training process and monitor the model's performance.


### Evaluating the Best Model Checkpoint

To evaluate the best model checkpoint obtained during the training process, you can use the following command:

```bash
PYTHONPATH=. python examples/evaluate.py
```

Executing this command runs the `evaluate.py` script, which performs an evaluation on the best model checkpoint.
This evaluation can include various metrics and measures to assess the model's performance on a specific task or dataset.
By using this example, you can examine the effectiveness of the trained model and obtain insights into its capabilities.

Feel free to explore and modify these example scripts to suit your needs.
You can experiment with different configurations, architectures, and hyperparameters of neural networks to better understand their behavior and performance.
MLnumpy provides a flexible and customizable framework that encourages you to create and test your own scripts, enabling you to dive deeper into the world of neural networks and machine learning.


## Contribution

Contributions are very welcome.
Tests can be run with [tox](https://tox.wiki/en/latest/), please ensure the coverage at least stays the same before you submit a merge request.


## License

Distributed under the terms of the [MIT](https://opensource.org/license/mit/) license, MLnumpy is free and open source software.


## Issues

If you encounter any problems, please email me at <mateusz.baran.sanok@gmail.com>, along with a detailed description.
