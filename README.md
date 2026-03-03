# Hand Gesture Image Collection & Labeling (Computer Vision)

This project collects hand gesture images using a webcam and prepares them for object detection training with LabelImg.

---

# Features

- Webcam image capture  
- Automatic folder creation for labels  
- LabelImg integration  
- OS-based setup (Windows / Linux / macOS)  
- Dataset packaging for training  

---

# Libraries Used

- OpenCV  
- os  
- uuid  
- subprocess  
- time  

---

# Labels

Default gesture labels:
- thumbsup  
- thumbsdown  
- thankyou  
- livelong  

---

# Workflow

1. Create folders for each label  
2. Capture images from webcam  
3. Save images with unique filenames  
4. Install and run LabelImg  
5. Annotate images  
6. Split dataset into train and test  
7. Archive dataset  

---

# How to Run

```bash
python collect_and_label.py
