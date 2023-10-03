# Real-Time Sign Language Recognition with TensorFlow and MediaPipe Holistic
# Overview
This project aims to create a real-time sign language recognition model using TensorFlow and MediaPipe Holistic. 
The model takes key landmarks points extracted from face, pose, and hands using MediaPipe Holistic and treats them as if they were audio spectrogram data. It employs a CNN model, such as ResNet, for classifying sign language gestures.
# Dataset
The dataset used for training and evaluation consists of key landmarks points extracted using MediaPipe Holistic.

# Data Preprocessing
To prepare the data for training, the following preprocessing steps were performed:

1. Key Landmarks Extraction: Extracted key points include those for eyes, nose, shoulders, and hands, resulting in a total of 80 data points per sample..
2. Standardization: Standardized the values of the extracted key landmarks to have zero mean and unit variance.
3. Time Axis Interpolation: Interpolated the time axis to a fixed size of 160 using 'nearest' interpolation to ensure consistent data input size.
4. Tensor Formation: Obtained a tensor with a size of 160x80x3 to be used as input for the CNN model.
5. Data Augmentation: Implemented various image processing data augmentation techniques during training to improve model generalization.
![image](https://github.com/NaniNorris/Real-time-sign-language-detection/assets/111329357/0aeb9492-2500-4aa9-95c4-4c343f0d3fcb)

# Model Architecture
The model architecture employed for sign language recognition is a Convolutional Neural Network (CNN), possibly based on ResNet or a similar architecture.
#
![image](https://github.com/NaniNorris/Real-time-sign-language-detection/assets/111329357/aec61826-f06e-4a55-9aa0-0ff215fda803)



