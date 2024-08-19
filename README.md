# Understanding Convolutional Neural Networks (CNNs)

## What is a Convolutional Neural Network?

A Convolutional Neural Network (CNN) is a type of neural network effective for processing data that has a grid-like topology, such as images. Unlike traditional neural networks, CNNs automatically and adaptively learn spatial hierarchies of features from input images. They are composed of multiple layers, each designed to extract specific features from the data, such as edges, textures, and complex shapes.

### Key Components of CNNs

1. **Convolutional Layer:**
   - The core building block of a CNN.
   - Applies a set of filters (kernels) to the input data to produce a feature map.
   - Captures spatial relationships by considering the local regions of the image.

2. **Pooling Layer:**
   - Reduces the dimensionality of feature maps while retaining important information.
   - Common pooling operations include max pooling and average pooling.
   - Helps in making the model more robust to translations and distortions in the image.

3. **Activation Function:**
   - Introduces non-linearity into the model, allowing it to learn complex patterns.
   - The most commonly used activation function in CNNs is the ReLU (Rectified Linear Unit).

4. **Fully Connected Layer:**
   - Connects every neuron in one layer to every neuron in the next layer.
   - Typically used at the end of the network for classification tasks.
   - The output is usually passed through a softmax function to generate class probabilities.

5. **Dropout Layer:**
   - A regularization technique used to prevent overfitting.
   - Randomly drops neurons during the training process, which forces the network to learn more robust features.

## How Does a CNN Work?

When an image is passed through a CNN:

1. **Convolution:** The image is filtered using a set of convolutional filters, creating feature maps that highlight various aspects of the image, like edges or textures.
2. **Pooling:** The feature maps are downsampled to reduce their size, retaining the most important information while reducing computational complexity.
3. **Flattening:** The pooled feature maps are flattened into a one-dimensional vector, which is then fed into fully connected layers.
4. **Classification:** The fully connected layers perform the classification, outputting probabilities for each class.

### Example Workflow:

1. **Input Image:** Consider an input image of a cat.
2. **Convolutional Layers:** The CNN applies filters that detect edges, textures, and shapes specific to a cat.
3. **Pooling Layers:** The feature maps are downsampled to focus on the most prominent features.
4. **Fully Connected Layer:** The extracted features are used to classify the image, resulting in a high probability for the "cat" class.

## Why Use a CNN?

### Advantages of CNNs:

- **Automatic Feature Extraction:** Unlike traditional methods, CNNs do not require manual feature extraction, as they learn to identify features during training.
- **Translation Invariance:** Pooling layers help CNNs recognize objects regardless of their position in the image.
- **Parameter Sharing:** Convolutional layers share parameters, reducing the number of parameters in the model, which makes CNNs more efficient and easier to train.
- **Effective for Image Data:** CNNs are specifically designed to work well with image data, making them ideal for tasks like image classification, object detection, and image segmentation.

### Applications of CNNs:

- **Image Classification:** Recognizing objects in images (e.g., cats, dogs, cars).
- **Object Detection:** Identifying and locating objects within an image.
- **Image Segmentation:** Partitioning an image into different regions based on features.
- **Medical Imaging:** Analyzing medical scans, such as MRI or CT images, for diagnosis.

### When to Use a CNN?

You should consider using a CNN when you are working with image data or any type of data that has a spatial structure, and you need a model that can automatically learn and identify patterns, features, and hierarchical relationships within the data.

## CNN project walkthrough
### Reading MNIST Data
   1. First of all we need to add the data that is going to be read, in this case we use two csv files, one for training and one for testing.
   2. Create the "DataReader" class that will extract the information (numbers) from the test/train file and store it in the "Image" class as a matrix and the label of the number.
   3. Also for the "Image" class we override the "toString" method so we can see what image is stored in the list.

### Add The Abstract Layer Class
   - Add the "Layer" class (abstract) so we have the template for all the layers that this network will have.

### Fully Connected Layer
   - The first layer where there is the definition of the back propagation and forward pass.
      - The back propagation uses the losses of the calculations so the training process can be more accurate.

      