# Convolutional 2D Deep Neural Network for Galaxy Classification

This is a **Convolutional Deep Neural Network** made from scratch for image classification based on a **one-hot encoding** technique for 4 possible categories. The neural network was trained on data scratched from **Galaxy Zoo** by **Zooniverse**.

## Model Architecture

The model is a **Sequential** model, each layer has 1 input tensor and 1 output tensor. The first layer is an **Input** layer, with the shape of the images, 128x128 (x=128, y=128), and 3 color channels (RGB):

$$X \in \mathbb{R}^{128 \times 128 \times 3}$$

There are then two convolutional layers (**Conv2D**) with **ReLU** activation, each followed by max-pooling layers:

$$Z_1 = \text{ReLU}(\text{Conv2D}(X, W_1, b_1, \text{strides}=2))$$

$$P_1 = \text{MaxPooling2D}(Z_1, \text{poolsize}=(2, 2), \text{strides}=2)$$

$$Z_2 = \text{ReLU}(\text{Conv2D}(P_1, W_2, b_2, \text{strides}=2))$$

$$\text{ReLU}(x) = max(0,x)$$

Where $W_1$ and $b_1$ are the weights and biases of the first convolutional layer and $W_2$ with $b_2$ referring to the second convolutional layer.

Then I flatten the output to 1D, as I am going to deal with a **one-hot categorical output**. 

$$F = \text{Flatten}(P_2)$$

$$F \in \mathbb{R}^{392}$$

The expression above converts the 3D tensor into a 1D tensor.

Following the last layer, there's a **Dense** layer with 16 neurons and a **ReLU ** activation. 

$$D_1 = \text{ReLU}(W_3 \cdot F + b_3)$$

The next layer is a **Dense** layer, with 4 features as the output and with **softmax** as the activation for the output:

$$\hat{Y} = \text{softmax}(W_4 \cdot D_1 + b_4)$$

$$\text{softmax}(z_i) = \frac{e^z_i}{\sum_{z_i} e^z_j}$$


Where $W_4$ and ${b_4}$ are the weights and biases of the output dense layer, and $\hat{Y} \in \mathbb{R}^{4}$.


## Training

On the training phase, I use the **Adam** optimizer, which is widely used for machine learning, mainly because of its adaptative nature, in this project I use it to maintain the learning rate of the model. For the loss, I use for the loss function **CategoricalCrossentropy** as I'm dealing with a **one-hot enconded** dataset.

The model is trained in 12 epochs, the final benchmarks are:

```val_categorical_accuracy: 0.6821 - val_auc: 0.8834```

## Results

Running a test with 4 random galaxies from the dataset that has 1400 galaxies, the raw results are:

```
Galaxy_0
	Model prediction: [0.17770563 0.6630151  0.07265422 0.08662497]
	True label: Ringed (1)
	Correct: True
Galaxy_1
	Model prediction: [0.23947124 0.00664927 0.4942127  0.25966683]
	True label: Merger (2)
	Correct: True
Galaxy_2
	Model prediction: [0.22968078 0.10202686 0.3388059  0.32948646]
	True label: Merger (2)
	Correct: True
Galaxy_3
	Model prediction: [1.2136501e-04 7.2277919e-07 1.8424975e-02 9.8145294e-01]
	True label: Merger (2)
	Correct: False
Galaxy_4
	Model prediction: [0.04124738 0.03621183 0.16665128 0.75588953]
	True label: Merger (2)
	Correct: False
```

The following images present the visualization of the inferences, the image with black and white pictures represent the passage of the input through the layers, the big picture has the category predicted above written above it as the output:

### Galaxy 1

![image](https://github.com/user-attachments/assets/ed143ee0-3e23-4a2e-a0be-07cae18bc2f2)

![image](https://github.com/user-attachments/assets/70cd80d7-a3d0-4e94-b914-2982bb88573a)

![image](https://github.com/user-attachments/assets/af909bc3-3288-4cd7-8d7a-ac1403fceace)

### Galaxy 2

![image](https://github.com/user-attachments/assets/b5643947-16db-4b1f-9cf0-20006fb61c8b)

![image](https://github.com/user-attachments/assets/933b53b9-e919-40cb-b8d4-91321ec84e91)

![image](https://github.com/user-attachments/assets/5923237b-f7ce-43b1-ab1e-5fd041eec4f3)

### Galaxy 3

![image](https://github.com/user-attachments/assets/94e0c48f-5863-45ae-9498-2361010e27a3)

![image](https://github.com/user-attachments/assets/76c3122d-29e8-4827-b648-bc8875fd00dd)

![image](https://github.com/user-attachments/assets/52e5935b-19d9-4096-b0cc-19a71d3c7ab8)

### Galaxy 4

![image](https://github.com/user-attachments/assets/e2651409-9b86-4873-aae2-1235deee6345)

![image](https://github.com/user-attachments/assets/6091e201-226a-431c-ba0d-c3a430291afa)

![image](https://github.com/user-attachments/assets/3fe3cee5-4e94-4d63-811e-98e7650aa274)

## Conclusion

I love working with deep neural networks because of its complexity, specially when it's a convolutional one, that deals with images rather than plain texts or numbers. It has two datasets, images and labels, that are later separated into two subsets for each one: training and validation. This process has taught me a lot and has improved my mathematical skills in machine learning as well as polished my skills in the coding side when using **TensorFlow**, **Keras**, **Numpy** and **Scikit-Learn**.

Thanks for reading! :)
