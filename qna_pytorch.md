# Important Questions: Pytorch

### Question: Introduction to PyTorch

Can you explain what PyTorch is and why it is popular among deep learning researchers? How does it differ from other deep learning frameworks like TensorFlow?

### Answer:

PyTorch is an open-source machine learning library primarily developed by Facebook's AI Research lab. It's designed for deep learning and artificial intelligence applications, providing a flexible and intuitive framework for building and training neural networks.

PyTorch is popular among deep learning researchers for several reasons:

- **Dynamic Computational Graphs**: PyTorch uses a dynamic computational graph, allowing for more flexible model architectures and easier debugging. This is particularly useful for research, where models may need to be modified frequently.

- **Pythonic Interface**: PyTorch has a clean, intuitive API that closely resembles Python's native data structures and operations, making it easier for researchers to write and understand code.

- **Easy Debugging**: Because of its dynamic nature, PyTorch allows for line-by-line debugging, which is extremely helpful when developing complex models.
GPU Acceleration: PyTorch provides seamless support for GPU acceleration, essential for training large neural networks efficiently.

- **Rich Ecosystem**: It has a growing ecosystem of tools and libraries, including torchvision for computer vision tasks and torchaudio for audio processing.

- **Research-Friendly**: Many cutting-edge research papers in deep learning are implemented in PyTorch, making it easier for researchers to reproduce and build upon existing work.

#### Differences from TensorFlow:

- **Computational Graph**: The main difference is in how the computational graphs are defined. TensorFlow traditionally used static computational graphs (though this changed with TensorFlow 2.0), while PyTorch uses dynamic graphs.

- **Debugging**: PyTorch's dynamic graph makes debugging easier and more intuitive compared to TensorFlow's static graph approach.

- **Deployment**: TensorFlow has historically been stronger in production deployment scenarios, though PyTorch has been catching up in this area.

- **Community and Ecosystem**: TensorFlow has a larger overall ecosystem and is more widely used in industry, while PyTorch is more popular in the research community.

- **Learning Curve**: Many find PyTorch easier to learn due to its Pythonic nature, whereas TensorFlow (especially earlier versions) had a steeper learning curve.

Both frameworks are powerful and widely used, with the choice often coming down to specific project requirements, personal preference, and the ecosystem of tools and libraries available for each.


### Question: Tensors

What is a tensor in PyTorch, and how does it compare to a NumPy array? Can you create a 3x3 tensor filled with random numbers and explain the basic tensor operations you can perform on it?

### Answer:

A tensor in PyTorch is a fundamental data structure that represents a multi-dimensional array of numerical values. It's similar to a NumPy array but optimized for deep learning operations and GPU acceleration.

#### Comparison to NumPy arrays:

- **GPU Support**: PyTorch tensors can easily be moved to GPU for faster computations, while NumPy arrays are CPU-bound.
- **Automatic Differentiation**: PyTorch tensors support automatic gradient computation, essential for neural network training.
- **In-place Operations**: PyTorch provides more in-place operations to modify tensors without creating new ones.
- **Integration**: PyTorch tensors are deeply integrated with neural network modules and optimizers.

```python
import torch

# Create a 3X3 tesnor with random values from a uniform distribution betwen 0 and 1
random_tensor = torch.rand(3, 3)

print(random_tensor)
```

#### Basic Tensor Operations

Arithmatic Operations

```python
# Addition
result = random_tensor + 1

# Multiplication
result = random_tensor * 2

# Element-wise multiplication
result = random_tensor * random_tensor
```

Reshaping
```python
# Reshape to 1X9
reshape = random_tensor(1, 9)
```

Indexing and Slicing
```python
# Get the first row
first_row = random_tensor[0]

# Get a specific element
element = random_tensor[1, 1]
```

Aggregation

```python
# Sum all elements
total = rensom_tensor.sum()

# Mean of all elements
mean = random_tensor.mean()
```

Matrix Operations
```python
# Matrix Multiplication
result = torch.matmul(random_tensor, random_tensor)

# Transpose
transposed = reandom_tensor.t()
```

Changing data type

```python
# Convert to integer type
int_tensor = random_tensor.int()
```

Moving to GPU if available
```python
gpu_tensor = random_tensor.cuda()
```

These operations demonstrate the versatility of PyTorch tensors in handling various mathematical operations crucial for deep learning tasks.


### Question: Autograd

How does PyTorch's autograd system work? Can you provide an example of how to use autograd to compute the gradients of a simple mathematical function?

### Answer

PyTorch's autograd system is a key feature that enables automatic differentiation of tensor operations. It's crucial for training neural networks as it automatically computes gradients, which are essential for optimization algorithms like gradient descent.

#### How autograd works

- **Dynamic Computation Graph**: As you perform operations on tensors, PyTorch builds a dynamic computation graph behind the scenes.
- **Recording Operations**: Each tensor that requires gradient tracking has a requires_grad attribute set to True. PyTorch records all operations performed on these tensors.
- **Backward Pass**: When you call .backward() on a tensor, PyTorch automatically computes the gradients of all tensors in the computation graph with respect to that tensor.
- **Gradient Accumulation**: Gradients are accumulated in the .grad attribute of each tensor that requires gradients.

Here's an example of using autograd to compute gradients of a simple mathematical function:

```python
import torch
# Create tensors with requires_grad=True to track computation
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Define a simple function: f(x, y) = x^2 + y^3

z = x**2 + y**3

# Compute gradient
z.backward()

# Print the gradients
print("dz/dx :", x.grad)    # Should be 2 * x = 4
print("dz/dy : ", y.grad)   # should be 3 * y^2 = 27 
```

Let's break down what's happening:

We create two tensors `x` and `y` with `requires_grad=True`, telling PyTorch to track operations on these tensors.

We define our function `z = x^2 + y^3`.

When we call `z.backward()`, PyTorch automatically computes the gradients of `z` with respect to all tensors in its computation graph that have `requires_grad=True`.

The gradients are stored in the `.grad` attribute of each tensor:

For `x: dz/dx = 2x = 2 * 2 = 4`
For `y: dz/dy = 3y^2 = 3 * 3^2 = 27`


This simple example demonstrates the power of autograd. In practice, this system allows PyTorch to automatically compute gradients for complex neural networks with millions of parameters, making it possible to train these networks using gradient-based optimization algorithms.

#### Some additional points about autograd:

- **Gradient accumulation**: If you call `backward()` multiple times, gradients will accumulate. Use `tensor.grad.zero_()` to reset gradients between iterations.

- **no_grad() context**: You can use `with torch.no_grad()`: to temporarily disable gradient tracking, which is useful during evaluation to save memory and computation.

- **Higher-order gradients**: PyTorch supports higher-order gradients, allowing you to compute gradients of gradients.

- **Retain graph**: By default, the computation graph is freed after `backward()`. Use `retain_graph=True` if you need to call `backward()` multiple times on the same graph.

Understanding autograd is crucial for effectively using PyTorch, especially when implementing custom loss functions or working with complex model architectures.

### Question: Data Loading

How do you load and preprocess data in PyTorch? Can you explain the roles of Dataset and DataLoader classes and provide a simple example of loading an image dataset?

### Answer

#### Data Loading and Preprocessing in PyTorch

Roles of Dataset and DataLoader Classes

1. **Dataset Class**

The Dataset class in PyTorch represents a dataset. It is an **abstract class**, and you need to subclass it to define your own dataset. The Dataset class requires you to implement two methods:

`__len__()`: This method returns the number of samples in the dataset.

`__getitem__(index)`: This method returns the data sample at the given index. It is here that you typically load and preprocess the data.

2. **DataLoader Class**

The DataLoader class in PyTorch is responsible for loading data from a Dataset and provides an iterable over the dataset. It handles batching, shuffling, and loading the data in parallel using multiprocessing workers. The DataLoader class makes the process of loading data efficient and easy to use during training.

**Steps to Load and Preprocess Data**

- **Create a Custom Dataset Class**: Subclass the Dataset class and implement the required methods.

- **Instantiate the Dataset**: Create an instance of the custom dataset class.

- **Use DataLoader**: Pass the dataset instance to a DataLoader to handle batching, shuffling, and parallel loading.

Example: Loading an Image Dataset

Let's create an example of loading an image dataset using PyTorch. We'll use a simple custom dataset that loads images from a directory.

**Step 1: Import Libraries**

```python
import torch
from torch.utils.data import Datase, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Define the custom dataset class

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

# Define Transformation
transform = transforms.compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# Instantiate the Dataset and DataLoader
image_dir = 'path/to/ima/directory'
dataset = ImageDataset(image_dir=image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_Size=32, shuffle=True, num_workers=4)

# Iterate over the dataloader
for batch in dataloader:
    print(batch.shape)
```

### Question: Basic Model Creation

How do you define a simple neural network in PyTorch? Can you create a basic feedforward neural network with one hidden layer and explain each component?

### Answer

Defining a simple neural network in PyTorch involves using the `torch.nn` module to create layers and the `torch.optim` module to define an optimizer.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network class

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size output_size):
        super(SimpleNN, self).__init__()

        # Define the layers
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Forward pass through the network
        x = self.hidden_layer(x)    # Pass input through input layer
        x = self.activation(X)      # Apply activation function
        x = self.output_layer(x)    # Pass through output layer
        return x


# Instantiate the model
input_size = 784    # Example input size (e.g., 28x28 images flattened)
hidden_size = 128   # Number of neurons in the hidden layer
out_put_size = 10   # Number of output classes (e.g., for MNIST: digits 0-9)

model = SimpleNN(input_size, hidden_size, output_size)

print(model)

# Define Loss Function and Optimizer

criterion = nn.CrossEntropy()   # Loss function for classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer
```

### Question: Loss Functions

What are loss functions, and why are they important in training neural networks? Can you list some common loss functions used in PyTorch and explain in which scenarios you would use them?

### Answer

Loss functions, also known as cost functions or objective functions, **measure how well a neural network's predictions match the actual target values**. They are crucial in training neural networks because they provide a signal to guide the optimization process. By minimizing the loss function, the network learns to make better predictions.

#### Importance of Loss Functions

- **Guidance for Training**: Loss functions provide a quantitative measure of the network's performance, allowing the optimization algorithm to update the network's weights to improve accuracy.

- **Model Evaluation**: They help in evaluating how well the network is performing during training and testing phases.

- **Error Feedback**: Loss functions provide feedback in the form of gradients during backpropagation, enabling the network to learn from its mistakes.

Common Loss Functions in PyTorch and Their Scenarios

#### Mean Squared Error Loss (nn.MSELoss)

**Scenario**: Used for regression tasks where the goal is to predict a continuous value.

**Description**: Measures the average squared difference between the predicted values and the actual values.

**Usage**:

```python
loss_fn = nn.MSELoss()
```

#### Cross-Entropy Loss (nn.CrossEntropyLoss)

**Scenario**: Used for classification tasks where the target is a class index.

**Description**: Combines `nn.LogSoftmax` and `nn.NLLLoss` in one single class. It is suitable for multi-class classification problems.

**Usage**:

```python
loss_fn = nn.CrossEntropyLoss()
```

#### Binary Cross-Entropy Loss (nn.BCELoss)

**Scenario**: Used for binary classification tasks.

**Description**: Measures the binary cross-entropy between the predicted values and the actual binary labels.

**Usage**:

```python
loss_fn = nn.BCELoss()
```

#### Binary Cross-Entropy with Logits Loss (nn.BCEWithLogitsLoss)

**Scenario**: Used for binary classification tasks, combining a sigmoid layer and binary cross-entropy loss in one single class.

**Description**: More numerically stable than using a plain nn.BCELoss with a separate sigmoid layer.

**Usage**:
```python
loss_fn = nn.BCEWithLogitsLoss()
```

#### Negative Log Likelihood Loss (nn.NLLLoss)

**Scenario**: Used for classification tasks with already log-probabilized outputs (usually combined with `nn.LogSoftmax`).

**Description**: Measures the negative log likelihood of the predicted class probabilities.

**Usage**:

```python
loss_fn = nn.NLLLoss()
```

#### Kullback-Leibler Divergence Loss (nn.KLDivLoss)

**Scenario**: Used in scenarios where you want to measure how one probability distribution diverges from a second, expected probability distribution.

**Description**: Measures the Kullback-Leibler divergence between two distributions.

**Usage**:

```python
loss_fn = nn.KLDivLoss()
```

#### Huber Loss (nn.SmoothL1Loss)

**Scenario**: Used for regression tasks that are sensitive to outliers.

**Description**: Combines the best properties of `nn.L1Loss` and `nn.MSELoss`. It is less sensitive to outliers than `nn.MSELoss`.

**Usage**:

```python
loss_fn = nn.SmoothL1Loss()
```

Examples of Usage
Here's an example demonstrating how to use a loss function in a training loop:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate the model, loss function, and optimizer
model = SimpleModel(input_size=10, hidden_size=5, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example data
inputs = torch.randn(10)
targets = torch.randn(1)

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, targets)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'Loss: {loss.item()}')
```


### Question:  Optimizers

How do optimizers work in PyTorch? Can you compare different optimizers like SGD, Adam, and RMSprop, and discuss when you might choose one over the others?

### Answer

Optimizers in PyTorch are responsible for updating the parameters of a neural network based on the computed gradients during the training process. They implement various algorithms to adjust the learning rate and update rules, aiming to minimize the loss function efficiently. Let's dive into the details of some popular optimizers and compare them:

#### Stochastic Gradient Descent (SGD)

SGD is the most basic optimizer. It updates the parameters in the opposite direction of the gradient of the loss function with respect to the parameters.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

**Characteristics**:

Simple and memory-efficient

- Can be slow to converge, especially for ill-conditioned problems
- Learning rate is fixed and needs careful tuning
- Can oscillate around minima, especially with high learning rates

Use when:

- You have a simple, well-conditioned problem
- You want a baseline optimizer to compare against
- Computational resources are limited


#### Adam (Adaptive Moment Estimation)

Adam combines ideas from `RMSprop` and `momentum` optimization. It adapts the learning rate for each parameter using estimates of first and second moments of the gradients.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Characteristics**:

- Adapts learning rate for each parameter
- Incorporates momentum for faster convergence
- Generally works well out-of-the-box with default hyperparameters
- Computationally more expensive than SGD

**Use when**:

- You're working on a complex problem with a large dataset
- You don't want to spend much time tuning hyperparameters
- You need fast convergence


#### RMSprop (Root Mean Square Propagation)

RMSprop adapts the learning rates of the parameters using a moving average of squared gradients.

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
```

**Characteristics**:

- Adapts learning rate for each parameter
- Helps mitigate the vanishing gradient problem
- Generally performs well in non-stationary settings
- Can be sensitive to initial learning rate

**Use when**:

- You're working on recurrent neural networks
- You observe oscillations or slow convergence with SGD

#### Detailed Comparison:

**Adaptivity**:

- SGD: No adaptivity, uses a fixed learning rate.

- Adam: Adapts learning rates for each parameter using both first and second moments of gradients.

- RMSprop: Adapts learning rates using second moments of gradients.

**Momentum**:

- SGD: No inherent momentum (though a momentum variant exists).
- Adam: Incorporates momentum through the first moment estimate.
- RMSprop: No inherent momentum (though a momentum variant exists).

**Memory Requirements**:

- SGD: Lowest memory requirements.
- Adam: Highest memory requirements (stores two additional variables per parameter).
- RMSprop: Moderate memory requirements (stores one additional variable per parameter).


**Convergence Speed**:

- SGD: Generally slower, especially for ill-conditioned problems.
- Adam: Often converges faster, especially in early stages of training.
- RMSprop: Usually faster than SGD, can be competitive with Adam.


**Hyperparameter Sensitivity**:

- SGD: Highly sensitive to learning rate.
- Adam: Generally works well with default hyperparameters.
- RMSprop: Moderately sensitive to initial learning rate.

**Choosing an Optimizer**:

For new problems, start with Adam:

- It often performs well out-of-the-box and can quickly give you a sense of how your model is performing.
- If computational resources are limited, try SGD with momentum:
- It's memory-efficient and can perform well with proper tuning.

For recurrent neural networks, consider RMSprop or Adam:

- They handle non-stationary objectives well, which is common in RNNs.
- If Adam is converging too quickly to a suboptimal solution:
- Try SGD, which sometimes generalizes better, especially towards the end of training.

For very large models or datasets:

- Consider memory-efficient variants like AdamW or Adafactor.

For research purposes:

- Always compare against SGD as a baseline, as it's still competitive when well-tuned.

Remember, the choice of optimizer can significantly impact your model's performance, but it's not the only factor. Other considerations like learning rate schedules, regularization techniques, and proper initialization are also crucial for achieving optimal results.


### Question: Training Loop

Can you describe the typical steps involved in a PyTorch training loop? Write a basic training loop for a classification problem, explaining each step.

### Answer

A PyTorch training loop is the core component for training a neural network. It involves multiple steps that include loading data, forwarding the data through the network, computing the loss, backpropagating the error, and updating the model's parameters. 

Here’s a detailed breakdown of each step involved in a typical PyTorch training loop for a classification problem, along with a basic implementation.

Typical Steps in a PyTorch Training Loop

- **Prepare the Data**: Load the dataset and prepare data loaders for training and validation.
- **Define the Model**: Create an instance of the neural network model.
- **Define the Loss Function**: Choose an appropriate loss function for the task.
- **Define the Optimizer**: Select an optimizer to update the model's parameters.
- **Training Phase**:
    - **Loop over Epochs**: Iterate over the dataset multiple times.
    - **Loop over Batches**: Iterate over mini-batches of data.
    - **Forward Pass**: Compute the model's predictions.
    - **Compute Loss**: Calculate the loss using the loss function.
    - **Backward Pass**: Compute the gradients of the loss with respect to the model's parameters.
    - **Update Parameters**: Adjust the model's parameters using the optimizer.

- **Validation Phase**: Evaluate the model on a validation dataset to monitor performance and prevent overfitting.
- **Checkpointing**: Save the model periodically.

Basic Training Loop for a Classification Problem

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorData

# Prepare the data
X_train = torch.randn(100, 10)  # 100 sample points with 10 features each
y_train = torch.randint(0, 2, (100))    # 100 labels, binary classification

# Create a dataset and data loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the model

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
# Binary classification (2 classes)
model = SimpleNN(input_size=10, hidden_size=5, output_size=2)   

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
num_epochs = 20

for epoch in range(num_epoch):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)


        # Backward pass and optimization
        optimizer.zero_grad()   # Clear previous gradients
        loss.backward()         # Compute gradients
        optimizer.step()        #  Update the model's parameters using the computed gradients.

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], step [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f})
```

### Question: Custom Datasets and Transforms

How would you create a custom dataset in PyTorch for a dataset not included in standard libraries? Provide an example, including custom data transforms.

### Question: Transfer Learning

What is transfer learning, and how can it be implemented in PyTorch? Provide an example of how to fine-tune a pre-trained model on a new dataset.

### Question: Handling Imbalanced Data

How do you handle imbalanced datasets in PyTorch? Can you explain techniques like oversampling, undersampling, and using weighted loss functions with examples?

### Question: Advanced Model Architectures

How would you implement advanced neural network architectures such as ResNet or LSTM in PyTorch? Provide a brief overview and a code snippet for one of these models.

### Question: Distributed Training

What is distributed training, and how can it be implemented in PyTorch? Discuss the torch.nn.parallel module and how it facilitates distributed training.

### Question: Debugging and Profiling

How do you debug and profile a PyTorch model? What tools and techniques can you use to identify performance bottlenecks or issues in your model?

### Question: Quantization

What is model quantization, and why is it important? Can you describe the process of quantizing a PyTorch model and its benefits?

### Question: Mixed Precision Training

What is mixed precision training, and how does it benefit deep learning models? Provide an example of how to implement mixed precision training in PyTorch using torch.cuda.amp.

### Question: Dynamic vs. Static Computation Graphs

How does PyTorch's dynamic computation graph differ from static computation graphs used in frameworks like TensorFlow 1.x? What are the advantages and disadvantages of each?

### Question: Model Serialization

How do you save and load PyTorch models? Explain the difference between saving a model's state dictionary and saving the entire model, including an example of each method.

### Question: Deployment

How would you deploy a PyTorch model to a production environment? Discuss different deployment strategies and tools like TorchScript and ONNX.

### Question: Hyperparameter Tuning

How do you perform hyperparameter tuning in PyTorch? Can you discuss different techniques such as grid search, random search, and Bayesian optimization, and provide an example using a PyTorch model?

### Question: Explainability and Interpretability

How do you make a PyTorch model's predictions explainable and interpretable? Discuss techniques like Grad-CAM, SHAP, or LIME, and provide an example of applying one of these methods to a PyTorch model.

### Question: Creating Tensors

How do you create tensors in PyTorch? Provide examples of creating tensors from Python lists and NumPy arrays.

### Question: Tensor Operations

Explain basic tensor operations in PyTorch (e.g., addition, multiplication). How do you perform element-wise operations and tensor broadcasting?

### Question: Autograd in PyTorch

What is autograd in PyTorch? How does it enable automatic differentiation for computing gradients?

### Question: Defining Neural Networks

How do you define a neural network architecture using PyTorch's nn.Module? Provide an example of defining layers and forward propagation.

### Question: Loss Functions

Name and describe common loss functions used in PyTorch (e.g., nn.CrossEntropyLoss, nn.MSELoss). When would you use each?

### Question: Optimizer

What are optimizers in PyTorch? Describe the purpose of optimizers like torch.optim.SGD and torch.optim.Adam. How do you use them to update model parameters?

### Question: Training a Model

Explain the steps involved in training a neural network model in PyTorch. How do you iterate through batches of data and update weights using backpropagation?

### Question: Model Evaluation

How do you evaluate a trained model in PyTorch? Describe metrics such as accuracy, precision, recall, and F1-score.