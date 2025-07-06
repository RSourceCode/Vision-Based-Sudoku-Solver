# 🧩 Vision Based Sudoku Solver

An end-to-end computer vision pipeline to detect, extract, and optionally solve Sudoku puzzles from real-world images using a combination of **YOLOv8** for grid detection and a custom **LeNet-based digit classifier** for number recognition.

---

## 🔍 Overview

This project performs the following:

* **Detects Sudoku puzzles** in natural images using YOLOv8 object detection.
* **Applies perspective correction** to deskew and isolate the puzzle.
* **Segments the grid into 81 cells** and classifies digits using a trained CNN.
* **Solves the extracted board** using a classic backtracking algorithm.

---

## 📁 Project Structure

```
├── arial_narrow_7/                    # Font files (used to alternatively train the digit recognition model on a custom made dataset)
├── font-files/
├── helvetica-bold/
├── runs/                              # YOLO training and prediction outputs
├── SudokuDetection/                   # YOLOv8 dataset config (e.g., data.yaml)
├── Sudoku_venv/ (not uploaded)        # Python virtual environment (excluded due to size)
├── Complete_Solver.py                 # Main Sudoku detection + digit recognition script
├── tempCodeRunnerFile.py              # Temp VS Code file
├── Yolo_Sudoku_Box_Predictor_Model_T.py  # YOLO training script
├── yolo11n.pt                         # Trained YOLOv8n weights
├── MNIST_Classifier_04_07.pth         # Final LeNet digit classifier
├── MNIST_Classifier_*.pth / .torch    # Other classifier checkpoints
├── Test_sudoku.jpg / *.jpg            # Test images for inference
├── image.png                          # Possibly demo or visual resource
```

> ⚠️ **Note:** `Sudoku_venv/` (Python virtual environment) is excluded from version control due to its large size. Please use the installation guide below to recreate it locally.

---

## ⚙️ Technologies Used

| Tool / Library           | Purpose                                         |
| ------------------------ | ----------------------------------------------- |
| **YOLOv8 (Ultralytics)** | Detecting Sudoku grids in real-world images     |
| **PyTorch**              | Training and inference for digit classification |
| **OpenCV**               | Preprocessing, deskewing, and contour detection |
| **scikit-image**         | Cleaning cell borders and resizing              |
| **imutils**              | Perspective transform for grid extraction       |
| **NumPy**                | Array manipulations                             |

---

## 🧠 Model Details

### YOLOv8

* Used to detect the bounding box of the Sudoku grid.
* Configured using `SudokuDetection/data.yaml`.

### Digit Classifier (LeNet)

* A lightweight CNN inspired by LeNet-5.
* Trained on MNIST dataset.
* Handles grayscale, 28x28 input cells.
* Final model: `MNIST_Classifier_04_07.pth`.

---

## 🚀 How to Run

1. **Clone the repo**:

   ```bash
   git clone https://github.com/RSourceCode/VisionBasedSudokuSolver.git
   cd VisionBasedSudokuSolver
   ```

2. **Set up environment**:

   ```bash
   python -m venv Sudoku_venv
   source Sudoku_venv/bin/activate  # or Sudoku_venv\Scripts\activate on Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install torch torchvision torchaudio
   pip install ultralytics opencv-python scikit-image imutils numpy
   ```

4. **Ensure these files exist**:

   * `yolo11n.pt` – YOLOv8 weights
   * `MNIST_Classifier_04_07.pth` – LeNet digit model
   * `Test_sudoku.jpg` – Test image or replace with your own

5. **Run the main script**:

   ```bash
   python Complete_Solver.py
   ```

---

## 📊 Output

* The script prints a 9×9 Sudoku grid recognized from the input image.
* Example:

```
[[5, 3, 0, 0, 7, 0, 0, 0, 0],   
[6, 0, 0, 1, 9, 5, 0, 0, 0],   
[0, 9, 8, 0, 0, 0, 0, 6, 0],   
[8, 0, 0, 0, 6, 0, 0, 0, 3],   
[4, 0, 0, 8, 0, 3, 0, 0, 1],   
[7, 0, 0, 0, 2, 0, 0, 0, 6],   
[0, 6, 0, 0, 0, 0, 2, 8, 0],   
[0, 0, 0, 4, 1, 9, 0, 0, 5],   
[0, 0, 0, 0, 8, 0, 0, 7, 9]]

Solved Sudoku:

[5, 3, 4, 6, 7, 8, 9, 1, 2]
[6, 7, 2, 1, 9, 5, 3, 4, 8]
[1, 9, 8, 3, 4, 2, 5, 6, 7]
[8, 5, 9, 7, 6, 1, 4, 2, 3]
[4, 2, 6, 8, 5, 3, 7, 9, 1]
[7, 1, 3, 9, 2, 4, 8, 5, 6]
[9, 6, 1, 5, 3, 7, 2, 8, 4]
[2, 8, 7, 4, 1, 9, 6, 3, 5]
[3, 4, 5, 2, 8, 6, 1, 7, 9]
```

---

## ✅ Features

* Fully automated grid detection and digit recognition
* Works on real-world photos with perspective distortion
* Lightweight and fast
* Clean segmentation and digit recognition pipeline
* Solves Sudoku using a simple backtracking algorithm

---

## 🤝 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* [LeNet-5 Architecture](http://yann.lecun.com/exdb/lenet/)
* [OpenCV](https://opencv.org/)
* [scikit-image](https://scikit-image.org/)
