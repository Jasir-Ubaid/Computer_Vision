# Computer_Vision
# YOLOv4 Object Detection using OpenCV

## 📌 Project Overview
This project implements a robust Object Detection model using **YOLOv4 (You Only Look Once)** and **OpenCV's Deep Neural Network (DNN)** module. It takes static images, processes them through the YOLO network, and accurately identifies multiple objects within the frame, drawing bounding boxes and assigning confidence scores based on the COCO dataset (80 distinct classes).

## 🛠️ Tech Stack
* **Python 3.x**
* **OpenCV (`cv2`)**: For image processing and loading the DNN model.
* **NumPy**: For array manipulation and mathematical operations.
* **Matplotlib**: For rendering the output images and plotting data visualizations.
* **YOLOv4 Weights & Config**: Pre-trained model files for high-accuracy detection.

## 🚀 How to Set Up and Run

### 1. Install Dependencies
Make sure you have the required Python libraries installed:
`pip install opencv-python numpy matplotlib`

### 2. Running the Notebook
Open the Jupyter Notebook to run the project:
`jupyter notebook Object_Detection.ipynb`

### 3. What happens when you run it?
You do not need to download the heavy YOLO weights manually! The notebook is designed to be self-contained:
* It will automatically download `yolov4.weights`, `yolov4.cfg`, and `coco.names` if they are missing.
* It will download sample images (`dog.jpg`, `person.jpg`) to test the model.
* It will output the images with bounding boxes drawn over the detected objects and display charts showing confidence distributions.

## 🧠 How it Works
* **Preprocessing:** The image is converted into a "blob" (`cv2.dnn.blobFromImage`) and scaled down to 416x416 to match the YOLO network's expected input format.
* **Forward Pass:** The blob is passed through the network, outputting bounding boxes, class IDs, and confidence scores.
* **Non-Max Suppression (NMS):** `cv2.dnn.NMSBoxes` is applied to remove overlapping, redundant bounding boxes for the same object, keeping only the most accurate detection.
