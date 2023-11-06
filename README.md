# Ex-6-Handwritten Digit Recognition using MLP
## Aim:
To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
* Hardware – PCs
* Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:
* We have used an MLP to recognize the digits
* A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.
* A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.
* MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.
 
**MLP has the following features:**
   * Adjusts the synaptic weights based on Error Correction Rule
   * Adopts LMS
   * Possess Backpropagation algorithm for recurrent propagation of error
   * Consists of two passes
  	   * Feed Forward pass
	   * Backward pass
           
**Learning process** – Backpropagation

**Computationally efficient method**

![image 10](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)

**3 Distinctive Characteristics of MLP:**
   * Each neuron in network includes a non-linear activation function<br>
      ![image](https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png)

   * Contains one or more hidden layers with hidden neurons
   * Network exhibits high degree of connectivity determined by the synapses of the network

**3 Signals involved in MLP are:**
   * Functional Signal
   * input signal
   * propagates forward neuron by neuron thro network and emerges at an output signal
   * F(x,w) at each neuron as it passes

**Error Signal**

   * Originates at an output neuron
   * Propagates backward through the network neuron
   * Involves error dependent function in one way or the other
   * Each hidden neuron or output neuron of MLP is designed to perform two computations:
      * The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron
      * The computation of an estimate of the gradient vector is needed for the backward pass through the network

### **TWO PASSES OF COMPUTATION:**

* **In the forward pass:**

   * Synaptic weights remain unaltered

   * Function signal are computed neuron by neuron

   * Function signal of jth neuron is <br>
      ![image](https://user-images.githubusercontent.com/112920679/198814313-2426b3a2-5b8f-489e-af0a-674cc85bd89d.png)<br>
      ![image](https://user-images.githubusercontent.com/112920679/198814328-1a69a3cd-7e02-4829-b773-8338ac8dcd35.png)<br>
      ![image](https://user-images.githubusercontent.com/112920679/198814339-9c9e5c30-ac2d-4f50-910c-9732f83cabe4.png)
   * If jth neuron is output neuron, the m=mL  and output of j th neuron is<br>
      ![image](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)
   * Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer<br>
      ![image](https://user-images.githubusercontent.com/112920679/198814353-276eadb5-116e-4941-b04e-e96befae02ed.png)


* **In the backward pass,**

   * It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

   * It changes the synaptic weight by delta rule
      ![image](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)
* Gradient descent is used as an optimisation algorithm here.
* Gradient descent is an iterative first-order optimisation algorithm used to find a local minimum/maximum of a given function.

## Algorithm :
1. Import the necessary libraries of python.

2. After that, create a dataframe and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3. Divide the dataset into two parts. Where the first part is for training and the second is for testing.

4. Define all the basic functions needed to create an MLP.

5. Find the weights and bias of each neuon using the gradient descent algorithm.

6. Make predictions using the defined functions.

7. Create a function to test the predictions which also contains the algorithm to plot the image.

8. NOw, test the predictions and find the accuracy.

## Program
### Importing Libraries
```py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
```
### Splitting Dataset
```py
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) ## shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
Y_train
```
### Defining Basic Fuctions
```py

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
```
### Gradient Descent
```py
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
```
### Make Prediictions & Plot image
```py
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
```

### Test the predictions
```py
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
```
### Find the accuracy
```py
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
```

## Output
### Y_train 
![y](https://user-images.githubusercontent.com/93427237/204799233-3761fdc8-75d8-415e-8984-07f177ff0e61.jpeg)

### Gradient Descent
<img width=350 src=https://user-images.githubusercontent.com/93427237/204799290-a0d4c967-d385-4e70-8653-17ec681d1632.jpeg>
</br>
<img width=350 src=https://user-images.githubusercontent.com/93427237/204799266-0f468828-bb28-4ee2-a35f-947f98cfdb38.jpeg>


### Test Predictions
<img width=350 src=https://user-images.githubusercontent.com/93427237/204799339-7b373f72-ca87-4ea5-90e8-e31e74617139.jpeg>
</br></br>
<img width=350 src=https://user-images.githubusercontent.com/93427237/204799353-23eff6a2-ec2e-42ac-ae0d-751d70e72435.jpeg>
</br></br>
<img width=350 src=https://user-images.githubusercontent.com/93427237/204799386-0ccf717b-aa5c-4b7a-bae9-f808f2cba70e.jpeg>
</br></br>
<img width=350 src=https://user-images.githubusercontent.com/93427237/204799409-6aac310f-bd4b-46aa-8e4f-a98fc622c04e.jpeg>

### Accuracy
![acc](https://user-images.githubusercontent.com/93427237/204799318-9b430d32-2a1d-44dd-8a83-18cbd35473c0.jpeg)

## Result:
Thus, a MLP is created to recognize the handwritten digits
