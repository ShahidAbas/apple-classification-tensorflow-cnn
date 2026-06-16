# apple-classification-tensorflow-cnn

A deep learning project that classifies images of apples into **13 distinct varieties** using a custom Convolutional Neural Network (CNN) built from scratch with TensorFlow and Keras. This project marks my transition from classical Machine Learning into Deep Learning, moving from feature-engineered models to learning directly from raw image data.

## Dataset
- **Total images:** 6,405
- **Number of classes:** 13 apple varieties
- **Image size:** Resized to 100x100 pixels, RGB (3 channels)
- **Source:** Organized in a folder-per-class structure (one subfolder per apple variety) and loaded using `tf.keras.preprocessing.image_dataset_from_directory`
## Project Overview

The goal of this project is to take raw apple images and correctly classify them into one of 13 categories. The pipeline covers the full deep learning workflow: loading and visualizing image data, splitting it into train/validation/test sets, preprocessing and augmenting images, building and training a CNN, evaluating performance, and saving the final model for future inference and the classes are as follow:
* Apple Braeburn
* Apple Crimson Snow
* Apple Golden 1
* Apple Golden 2
* Apple Golden 3
* Apple Granny Smith
* Apple Pink Lady
* Apple Red 1
* Apple Red 2
* Apple Red 3
* Apple Red Delicious
* Apple Red Yellow 1
* Apple Red Yellow 2

## Model Architecture

The model is a custom CNN built with Keras's Sequential API:

- Built-in preprocessing layers: Resizing + Rescaling (normalizes pixel values to [0, 1])
- Built-in data augmentation: Random horizontal/vertical flips + random rotation (helps reduce overfitting)
- 5 × (Conv2D + MaxPooling2D) blocks — filter sizes: 32 → 64 → 64 → 64 → 64, all with 3x3 kernels and ReLU activation
- Flatten layer
- Dense layer (128 units, ReLU)
- Output Dense layer (13 units, Softmax)

Total trainable parameters: **~140,000** — a lightweight architecture by design, suited to a constrained 13-class problem.

**Compilation settings:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab (training environment, with dataset mounted from Google Drive)

## Training & Results

The model was trained for **10 epochs** on an 80/10/10 train/validation/test split (created with a custom dataset-partitioning function — see below).

| Metric | Value |
|---|---|
| Test Accuracy | **98.36%** |
| Test Loss | 0.0437 |

Training and validation accuracy/loss curves are plotted in the notebook to visually confirm the model converged well without significant overfitting.

## Custom Functions

This project includes two hand-written utility functions rather than relying purely on built-in helpers:

- **`get_dataset_partition_tf()`** — Splits a `tf.data.Dataset` into train, validation, and test sets using shuffle/take/skip, since TensorFlow datasets don't support array-style slicing the way NumPy arrays do.
- **`predict()`** — Wraps single-image inference: converts an image to an array, adds a batch dimension, runs the model, and returns the predicted class name along with a confidence score.

##  How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ShahidAbas/apple-variety-classification-cnn.git
   cd apple-variety-classification-cnn
   ```
2. Open `apple_classification.ipynb` in Google Colab or Jupyter Notebook.
3. Update the dataset path to point to your own apple image dataset (organized as one folder per class).
4. Run all cells to train the model and view evaluation results.

## Future Improvements

- Experiment with transfer learning (e.g., MobileNetV2, EfficientNet) to compare performance against the custom CNN
- Add a confusion matrix and per-class precision/recall for deeper evaluation
- Deploy the trained model behind a simple web app for live image classification

## License

This project is open-source under the MIT License. Feel free to use it for learning or as a reference for your own projects.
