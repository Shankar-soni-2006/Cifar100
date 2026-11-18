# CIFAR-100 Image Classification using CNN

A deep learning project that classifies images into 100 different categories using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The project uses the CIFAR-100 dataset to train an image classification model and demonstrates predictions on custom images.

## Overview

Image classification is a fundamental task in computer vision that involves identifying and categorizing objects within images.

This project implements a CNN-based image classifier capable of recognizing 100 different object categories from the CIFAR-100 dataset. It includes data preprocessing, model development, training, evaluation, and custom image prediction.

## Dataset

The project uses the **CIFAR-100 dataset**.

| Property | Details |
|---|---|
| Total Images | 60,000 |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Image Dimensions | 32 × 32 pixels |
| Color Channels | 3 (RGB) |
| Number of Classes | 100 |
| Dataset Format | NumPy arrays |

The dataset is loaded using TensorFlow's built-in CIFAR-100 dataset loader.

```python
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar100.load_data()
```

### Dataset Preprocessing

- Images are converted into floating-point values.
- Pixel values are normalized from the range `0–255` to `0–1`.
- Images retain their RGB color channels.
- Labels are represented as integer class indices.

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

## Model Architecture

The project implements a Sequential CNN architecture using TensorFlow Keras.

### Architecture Overview

| Layer | Configuration | Activation |
|---|---|---|
| Conv2D | 32 filters, 3 × 3 | ReLU |
| MaxPooling2D | 2 × 2 | — |
| Conv2D | 64 filters, 3 × 3 | ReLU |
| MaxPooling2D | 2 × 2 | — |
| Conv2D | 128 filters, 3 × 3 | ReLU |
| MaxPooling2D | 2 × 2 | — |
| Flatten | — | — |
| Dense | 256 neurons | ReLU |
| Dropout | 0.5 | — |
| Dense | 100 neurons | Softmax |

### Architecture Flow

```text
Input Image (32 × 32 × 3)
          │
          ▼
   Conv2D (32 Filters)
          │
          ▼
    Max Pooling
          │
          ▼
   Conv2D (64 Filters)
          │
          ▼
    Max Pooling
          │
          ▼
  Conv2D (128 Filters)
          │
          ▼
    Max Pooling
          │
          ▼
       Flatten
          │
          ▼
   Dense (256 Neurons)
          │
          ▼
    Dropout (0.5)
          │
          ▼
  Dense (100 Classes)
          │
          ▼
      Softmax
          │
          ▼
 Predicted Image Category
```

## Training Configuration

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Model | Sequential CNN |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |
| Epochs | 20 |
| Batch Size | 256 |
| Validation Split | 10% |
| Output Classes | 100 |

### Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Model Training

```python
model.fit(
    X_train,
    Y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.1
)
```

## Data Augmentation

The notebook defines a data augmentation pipeline using TensorFlow Keras.

The augmentation techniques include:

- Random Rotation
- Random Zoom
- Random Translation

```python
data_augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])
```

These transformations can help improve generalization by generating variations of training images.

> Note: The augmentation pipeline is defined in the notebook but is not currently applied inside the model training pipeline.

## Model Evaluation

The trained model is evaluated on the CIFAR-100 test dataset.

```python
loss, accuracy = model.evaluate(X_test, Y_test)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
```

Evaluation metrics:

- **Test Loss:** Measures the model's prediction error.
- **Test Accuracy:** Measures the percentage of correctly classified test images.

The notebook prints the test loss and test accuracy after training.

## Custom Image Prediction

The project also demonstrates predicting the class of a custom image.

The prediction pipeline includes:

1. Loading an image.
2. Converting it into RGB format.
3. Resizing it to 32 × 32 pixels.
4. Normalizing pixel values.
5. Adding a batch dimension.
6. Passing the image to the trained model.
7. Extracting the predicted class using `argmax`.

```python
img = Image.open(image_path).convert("RGB")
img = img.resize((32, 32))

img = np.array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

predict = model.predict(img)

predicted_class = np.argmax(predict)
print(class_names[predicted_class])
```

### Prediction Example

The notebook demonstrates custom image classification using a rose image.

The model predicts one of the 100 CIFAR-100 classes and displays the image with its predicted category.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network development |
| NumPy | Numerical operations and image preprocessing |
| Matplotlib | Image visualization |
| Pillow (PIL) | Image loading and resizing |
| Kaggle | Notebook environment |

## Repository Structure

```text
Cifar100/
│
├── cifar100.ipynb
└── README.md
```

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/Shankar-soni-2006/Cifar100.git
```

```bash
cd Cifar100
```

### Install Dependencies

```bash
pip install tensorflow numpy matplotlib pillow
```

### Run the Notebook

Open the notebook using Jupyter Notebook:

```bash
jupyter notebook
```

Alternatively, open the notebook in Google Colab or Kaggle Notebook and execute the cells sequentially.

## Future Improvements

- Integrate data augmentation directly into the training pipeline.
- Experiment with deeper CNN architectures.
- Implement transfer learning.
- Add confusion matrix visualization.
- Analyze class-wise accuracy.
- Improve model performance through hyperparameter tuning.
- Save and load trained models.
- Develop a web interface for real-time image classification.

## Author

**Shankar Soni**

- GitHub: [Shankar-soni-2006](https://github.com/Shankar-soni-2006)
- Repository: [CIFAR-100 Image Classification](https://github.com/Shankar-soni-2006/Cifar100)

---

*Built as a deep learning and computer vision project to explore CNN-based image classification.*
