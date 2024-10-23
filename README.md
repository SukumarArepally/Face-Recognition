#Tittle:Face Recognition with PCA and MLP

#Description:This project implements a face recognition pipeline using Principal Component Analysis (PCA) for dimensionality reduction and Multilayer Perceptron (MLP) for classification. The project is built using Python's sklearn library and utilizes the Labeled Faces in the Wild (LFW) dataset for training and testing.

#Table of Contents
'Installation


'Dataset

'Usage

'Results

'Contributing

'License

#Installation:

1.Clone the repository:
bash:
git clone https://github.com/yourusername/face-recognition-pca-mlp.git

2.Navigate to the project directory:
bash:
cd face-recognition-pca-mlp

3.Install the required dependencies:
bash:
pip install -r requirements.txt

#Dataset:

The project uses the Labeled Faces in the Wild (LFW) dataset. The dataset is loaded using fetch_lfw_people() from the sklearn.datasets module.

#Usage

Face Recognition Pipeline

Data Preparation:
The LFW dataset is fetched and split into training and test sets using train_test_split.

Dimensionality Reduction:
PCA is applied to reduce the dimensionality of the images, projecting the face images into a lower-dimensional space.

Classification:
An MLPClassifier is trained on the reduced features from PCA to classify the faces.

#How to Run

To run the face recognition pipeline:

bash:
python face_recognition.py

Plotting the Results:

The plot_gallery() function helps visualize the input images in a grid format. It takes in image data and displays them with appropriate titles.

#Results:

PCA is used to reduce the dimensionality of face images.
MLP successfully classifies the faces after dimensionality reduction.
Visualization of faces is available using matplotlib.

#Contributing:

Contributions are welcome! Please submit a pull request or open an issue if you have suggestions or improvements.

#License:

This project is licensed under the MIT License.
