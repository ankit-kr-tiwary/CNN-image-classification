# Chest X-ray Classification For Pneumonia Predictions

This project is a web application built with Flask that utilizes TensorFlow to classify chest X-ray images as either pneumonia or normal. The application allows users to upload images and receive predictions based on a trained CNN model.

## Project Overview

This application analyzes chest X-ray images to determine whether a patient has pneumonia or not. The model was trained on the Chest X-ray dataset from Kaggle, which contains thousands of labeled X-ray images. The trained model achieves an accuracy of **90.10%** in distinguishing between normal lungs and those affected by pneumonia.

## Live Application

You can access the live application deployed on Render at:
[https://chest-xray-classifier.onrender.com](https://cnn-image-classification-3pvh.onrender.com)

## Technologies Used

- **Flask**: For creating the web application and handling user interactions
- **TensorFlow/Keras**: For building and training the CNN model used in image classification
- **Google Colab**: Used for model training to leverage GPU resources
- **NumPy**: For numerical computations
- **Render**: For deploying the Flask application
- **Gunicorn**: For production deployment


## Project Structure

The project is organized with the following directory structure:

```
Projects/
|--- app.py
|--- index.html
|--- static/
|    |--- style.css
|---model_xray1.h5
|--- requirements.txt
```


## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/chest-xray-classifier.git
```

2. Navigate to the project directory:

```
cd Projects
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the application:

```
python app.py
```


## Usage

1. Open a web browser and navigate to `https://cnn-image-classification-3pvh.onrender.com`
2. Upload a chest X-ray image using the form on the page.
3. The application will display whether the X-ray indicates "Pneumonia" or "Normal."

## Model Training

The CNN model was trained using Google Colab to leverage its GPU resources. The Chest X-ray dataset from Kaggle was used, which contains labeled X-ray images categorized as "Pneumonia" or "Normal."

### Model Architecture

The model consists of the following layers:

1. **Conv2D Layers**: Extract features from images
2. **MaxPooling2D Layers**: Downsample feature maps to reduce dimensionality
3. **Dense Layers**: Fully connected layers for classification
4. **Dropout Layers**: Prevent overfitting by randomly dropping units during training

### Training Parameters

- **Epochs**: 25
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy


## Deployment

The application is deployed on Render, which provides a simple and efficient way to host web applications. The deployment process involves:

1. Pushing the code to a GitHub repository
2. Connecting the repository to Render
3. Configuring the build settings
4. Deploying the application

## Requirements

The required Python packages are listed in `requirements.txt`:

```
Flask
tensorflow
numpy
gunicorn==20.0.4
matplotlib
scikit-learn

```
## Acknowledgments

Special thanks to [Paul Mooney's Chest X-ray Dataset on Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) for providing the dataset used in this project.



