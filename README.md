# fashion-mnist-image-classifier
A deep learning project that uses TensorFlow and Keras to classify clothing images into 10 categories using the Fashion-MNIST dataset, with image preprocessing, model training, evaluation, and prediction on custom images.



# FASHION-MNIST IMAGE CLASSIFICATION

## Project Overview

This project explores clothing image classification using a neural network built with TensorFlow and Keras. It includes a model-training workflow and a separate prediction workflow for applying the saved model to custom clothing images.

The model learns to recognize clothing items from grayscale images and classify them into 10 categories. The prediction workflow also returns the probability scores for each clothing category and the predicted class.

## Problem Statement

Clothing image classification is a computer vision task that involves identifying clothing categories from image data. Automatically recognizing clothing items can be useful in applications such as fashion product categorization, image organization, and visual search.

This project uses the Fashion-MNIST dataset to develop a neural network that learns image patterns and predicts the corresponding clothing category.

## Dataset

The training notebook loads the Fashion-MNIST dataset using TensorFlow Datasets:

`fashion_mnist`

The dataset contains:
- 60,000 training images.
- 10,000 testing images.
- 28 × 28 pixel grayscale images.
- 10 clothing categories.

The clothing categories include T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

The dataset is downloaded through TensorFlow Datasets and is not required as a separate CSV file.

## Project Workflow

1. Model training
   - Load the Fashion-MNIST dataset using TensorFlow Datasets.
   - Retrieve dataset information and separate training and testing data.
   - Create a validation subset from the training dataset.
   - Normalize image pixel values to the range 0–1.
   - Shuffle and batch the training and validation datasets.
   - Build a Sequential neural network using a Flatten layer, two Dense hidden layers with 50 neurons each, and a 10-neuron Softmax output layer.
   - Compile the model using the Adam optimizer, Sparse Categorical Crossentropy loss, and accuracy metric.
   - Train the model for 5 epochs.
   - Evaluate the trained model using the test dataset.
   - Save the trained model as `fashion_model.keras`.

2. Prediction
   - Import `FashionMNISTPredictor` from `fashion_model.py`.
   - Load the saved model from `fashion_model.keras`.
   - Load an individual clothing image using its file path.
   - Convert the image to grayscale and resize it to 28 × 28 pixels.
   - Normalize pixel values to the range 0–1.
   - Generate probability scores for all 10 clothing categories.
   - Identify the predicted clothing category using the highest probability.
   - Return the image filename, predicted category, and winning probability.

## Technologies

- Python
- Jupyter Notebook
- TensorFlow
- Keras
- TensorFlow Datasets (TFDS)
- NumPy
- Pandas
- Neural Networks
- Softmax Classification

## Model Performance

The model was trained for 5 epochs and evaluated on the Fashion-MNIST test dataset.

| Metric | Result |
|---|---:|
| Training accuracy | 87.95% |
| Validation accuracy | 87.97% |
| Test accuracy | 86.23% |
| Test loss | 0.3803 |

## Repository Contents

- `Untitled(5).ipynb`: Original notebook for dataset loading, preprocessing, validation splitting, neural network development, model training, evaluation, and saving the trained model.
- `fashion_model.py`: Python module containing `FashionMNISTPredictor` for loading, preprocessing, and predicting custom clothing images.
- `Untitled2(1).ipynb`: Notebook demonstrating how to import and use `FashionMNISTPredictor` with the saved model to classify individual clothing images.
- `fashion_model.keras`: Saved trained neural network expected by the prediction notebook.
- `Pixelated Grayscale Running Sneaker.png`: Custom sneaker image used for prediction.
- `Pixelated Gray T-Shirt on Black.png`: Custom T-shirt image used for prediction.
- `Minimalist Gray Jeans on Black.png`: Custom trousers image used for prediction.
- `README.md`: Project documentation.

## How to Run

1. Clone or download the repository.
2. Install Python and the required packages:

   ```bash
   pip install numpy pandas tensorflow tensorflow-datasets scikit-learn matplotlib pillow jupyter
   ```

3. Ensure these files are in the project directory:
   - `fashion_model.py`
   - `fashion_model.keras` (for prediction)
   - `Untitled(5).ipynb`
   - `Untitled2(1).ipynb`
   - The custom clothing images used in the prediction notebook.

4. If `fashion_model.keras` does not exist, run the training notebook `Untitled(5).ipynb` to create it.
5. Start Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `Untitled2(1).ipynb` and run its cells to load the saved model and generate predictions.

## Prediction Example

```python
from fashion_model import FashionMNISTPredictor

predictor = FashionMNISTPredictor('fashion_model.keras')

predictor.load_and_clean_data(
    'Pixelated Grayscale Running Sneaker.png'
)

probabilities = predictor.predicted_probability()
categories = predictor.predicted_output_category()
results = predictor.predicted_outputs()

print(probabilities)
print(categories)
print(results)
```

The returned DataFrame contains:

- `Image`: Name or path of the input image.
- `Prediction`: Predicted clothing category.
- `Probability`: Probability score of the predicted category.

## Prediction Examples

The prediction notebook demonstrates the following results:

| Input image | Predicted category | Probability |
|---|---|---:|
| Running Sneaker | Sneaker | 78.64% |
| Gray T-Shirt | Shirt | 68.13% |
| Gray Jeans | Trouser | 97.70% |

## Saved Model

- `fashion_model.keras`: Trained neural network used for custom clothing image classification.

The prediction workflow depends on the saved model having the same architecture and class ordering used during training.

## Important Notes and Future Improvements

- The model was trained on Fashion-MNIST grayscale images with dimensions of 28 × 28 pixels.
- Custom images are resized and converted to grayscale before prediction.
- Real-world images may differ from Fashion-MNIST images in background, lighting, position, and appearance, which can affect prediction accuracy.
- The custom T-shirt image was classified as Shirt rather than T-shirt/top in the recorded prediction result.
- The model's probability output represents its confidence score, not a guarantee that the prediction is correct.
- Consider experimenting with Convolutional Neural Networks (CNNs) to improve image classification performance.
- Evaluate additional model architectures and training configurations.
- Consider developing a web interface for uploading clothing images and displaying predictions.

## Disclaimer

This is an educational machine-learning project developed for learning and experimentation with image classification, neural networks, TensorFlow, and Keras. 
Predictions on custom images are not guaranteed to be accurate and should not be treated as a production-ready image recognition system.
