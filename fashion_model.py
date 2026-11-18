"""Inference helper for the 28x28 grayscale Fashion-MNIST Keras model."""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf


class FashionMNISTPredictor:
    CLASS_NAMES = (
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    )

    def __init__(self, model_file='fashion_model.keras'):
        self.model = tf.keras.models.load_model(model_file)
        self.data = None
        self.image_files = None

    def load_and_clean_data(self, image_files):
        """Load a single image path or a list of paths, returning (N, 28, 28, 1)."""
        if isinstance(image_files, (str, Path)):
            image_files = [image_files]
        image_files = [str(p) for p in image_files]
        if not image_files:
            raise ValueError('Provide at least one image file.')

        images = []
        for path in image_files:
            image = tf.keras.utils.load_img(
                path, color_mode='grayscale', target_size=(28, 28)
            )
            array = tf.keras.utils.img_to_array(image).astype('float32') / 255.0
            images.append(array)

        self.data = np.stack(images, axis=0)
        self.image_files = image_files
        return self.data

    def predicted_probability(self):
        """Return probabilities for all 10 classes, shape (N, 10)."""
        if self.data is None:
            raise ValueError('Call load_and_clean_data() first.')
        return self.model.predict(self.data, verbose=0)

    def predicted_output_category(self):
        """Return the predicted clothing name for each image."""
        probabilities = self.predicted_probability()
        indices = np.argmax(probabilities, axis=1)
        return [self.CLASS_NAMES[i] for i in indices]

    def predicted_outputs(self):
        """Return image filename, predicted class, and winning probability."""
        probabilities = self.predicted_probability()
        indices = np.argmax(probabilities, axis=1)
        return pd.DataFrame({
            'Image': self.image_files,
            'Prediction': [self.CLASS_NAMES[i] for i in indices],
            'Probability': probabilities[np.arange(len(indices)), indices],
        })
