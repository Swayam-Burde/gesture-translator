# Project Report: Gesture Translator via Feature Engineering

**Course**: Computer Vision  
**Semester**: Spring 2026

---

## 1. Problem Statement & Relevance

Most modern object detection and gesture recognition systems rely on massive, slow deep learning computer vision models (such as YOLOv8 or large CNNs). While highly accurate, these are often computationally expensive to run on edge devices, require GPUs for inference, and function largely as "black boxes" where one cannot easily explain *why* top decisions were made.

The problem selected for this challenge was to build a tailored, lightweight gesture translator capable of accurately recognizing specific signs. 
This is highly relevant to this course because it bridges the gap between raw Computer Vision (tracking hands in visual space) and Data Science (tabular data classification), demonstrating that complex visual problems can often be solved gracefully with structured geometric feature engineering rather than brute-force neural networks.

## 2. Approach and Methodology

Our primary architecture moves away from raw pixel-based classification.

1. **Landmark Extraction**: Using MediaPipe, we extract the 3D (x, y, z) coordinates of 21 joints in the human hand per frame.
2. **Feature Engineering**: Instead of utilizing coordinates directly (which ruins translation and scale invariance), custom geometric functions were written in `src/feature_extraction.py`:
   - **Euclidean Distances**: Distances between adjacent fingertips, thumb tips to other fingertips, and all tips to the wrist.
   - **Scale Normalization**: All distances were scaled proportionally to palm length (Wrist to Middle Finger MCP) so the camera distance does not break the model.
   - **Angles**: Vector mathematics (`np.arccos(cosine_angle)`) was utilized to grab the exact bending angle at the PIP joints of every finger to determine folded vs. extended fingers.
3. **Classical Classification**: A scikit-learn Random Forest model is fitted onto this 17-dimensional vector per frame.

## 3. Key Decisions

- **Random Forest over SVM**: A Random Forest Classifier was chosen primarily because of its `feature_importances_` attribute. Due to our extensive geometric feature engineering, being able to verify *which* mathematical primitives actually helped the model accurately predict the gesture is highly valuable.
- **Dropping Raw Coordinates**: Attempting to feed raw X, Y, Z locations caused immediate model overfitting because the model memorized the screen location of the hand rather than the gesture. Abstracting positional tracking into internal relative vectors decoupled the hand state from pixel space.

## 4. Challenges Faced

1. **Numerical Instability**: When designing the 3D angle calculators, identical vectors triggered Zero Division Errors or issues due to Python's handling of floating-point inaccuracies around -1.0 and 1.0 cosines. We had to use `np.clip` and proper absolute norm checking to prevent the feature pipeline from silently crashing.
2. **Scale Variance**: Initial tests showed accuracy plummeted when stepping back from the camera. This required adding a dedicated normalizer—dividing all extracted distance features by the length of the user's physical palm tracking landmarks.

## 5. What Was Learned

This project clearly proved that Feature Engineering acts as an extreme data compression method. By algorithmically condensing a 640x480x3 RGB image into exactly 17 carefully engineered numbers, we bypassed the need for millions of parameters down to a tree-depth of fewer than 50. It served as a powerful reminder that defining a problem geometrically is often far superior to throwing raw compute at it. 
