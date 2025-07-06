from ultralytics import YOLO
import cv2
import torch
import torch.nn as nn
import numpy as np
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from skimage.transform import resize

## Defing my Neural Network : LeNet
class MNIST_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.flatten = nn.Flatten()  # tried start_dim=0
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # self.softmax = nn.Softmax()
        
    def forward(self, input):
        # print(input)
        x = self.conv1(input)
        # print(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x)
        x = self.flatten(x)
        # print(x)
        x = self.fc1(x)
        # print(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x
    

## Read the image and load the models
image = cv2.imread("Test_sudoku.jpg")
yolo_model = YOLO("runs/detect/train4/weights/best.pt")
_MNIST_Classifier = MNIST_Classifier()
_MNIST_Classifier.load_state_dict(torch.load("MNIST_Classifier_04_07.pth", map_location = torch.device('cpu')))
_MNIST_Classifier.eval()

## Detect Sudoku in the image from YOLO
results = yolo_model.predict(image, conf=0.5) # conf in the confindence threshold
xyxy = np.array(results[0].boxes.xyxy[0])
xyxy.astype(np.int32)
cropped_img = image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]


## Preprocess Image
gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
thres_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
bitwise_not_thres_img = cv2.bitwise_not(thres_img)


## Find Countors
contours, hierarchy = cv2.findContours(bitwise_not_thres_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)



## From the extracted contours, extract the suitable contour for Sudoku
sudokuContourPoints = None
for c in sorted_contours:
    perimeter = cv2.arcLength(c, True)
    approx_poly = cv2.approxPolyDP(c, 0.02 * perimeter, closed=True)
    if(len(approx_poly) == 4):
        sudokuContourPoints = approx_poly
        break

## If sudoku if available, perform Four Point transform to Deskew the sudoku.
if sudokuContourPoints is None:
    raise Exception("Couldn't find Sudoku in the image, please try again")


sudoku_img = four_point_transform(cropped_img, sudokuContourPoints.reshape(4, 2))
sudoku_img = cv2.cvtColor(sudoku_img, cv2.COLOR_BGR2GRAY)
# cv2.drawContours(cropped_img, [sudokuContourPoints], 0, (255, 255, 0), 3)


## Extract each cell
extracted_cells = []
cell_h = sudoku_img.shape[0] // 9
cell_w = sudoku_img.shape[1] // 9

for i in range(9):
    cell_img_row = []
    for j in range(9):
        cell = sudoku_img[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
        cell = clear_border(np.squeeze(cell < 150))
        cell = resize(cell, (280, 280))
        cell = resize(cell, (28, 28))
        # cv2.imshow(f"({i},{j})", cell.astype(np.float32))
        # cv2.waitKey(0)
        cell_img_row.append(cell)
    extracted_cells.append(cell_img_row)


## Check if the cell contains digit, if it does, use the MNIST_Classifier to get the digt from the image
sudoku = []
for i in range(9):
    sudoku_digit_row = []
    for j in range(9):
        cell = extracted_cells[i][j]
        (h, w) = cell.shape
        percentage_filled = np.count_nonzero(cell) / float(h * w)
        if(percentage_filled < 0.03):
            sudoku_digit_row.append(0)
        else:
            with torch.no_grad():
                # cv2.imshow("CELL", cell.astype(np.float32))
                # cv2.waitKey(0)
                # print(cell)
                cell = torch.from_numpy(cell.astype(np.float32))
                # print(cell)
                cell = cell[None, None, :, :]
                # print(cell)
                prediction = _MNIST_Classifier(cell)
                # print(prediction)
                _, digit = torch.max(prediction, 1)
                # print(digit)
                sudoku_digit_row.append(digit.item())
    sudoku.append(sudoku_digit_row)
    print(sudoku_digit_row)

## Helper function to check if the value put is a valid/Solved one or not
def isValid(sudoku, i, j, k):
    for l in range(9):
        if(sudoku[i][l] == k): 
            return False
        if(sudoku[l][j] == k): 
            return False
        
    for l in range(3):
        for m in range(3):
            if(sudoku[l + (i // 3) * 3][m + (j // 3) * 3] == k): 
                return False
    return True


def Solved(sudoku):
    for i in range(9):
        for j in range(9):
            if(sudoku[i][j] == 0):
                return False
            val = sudoku[i][j]
            sudoku[i][j] = 0 
            if(not (isValid(sudoku, i, j, val))):
                 return False
            sudoku[i][j] = val
    return True

## Finally after getting the sudoku from the Image, solve it using Back Tracking.
global _break
_break =  False
def SudokuSolver(sudoku, i = 0, j = 0):   
    global _break

    if(j == 9):
        j = 0
        i = i + 1

    if(i == 9):
        if(Solved(sudoku)):
            _break = True
        return  
    if(sudoku[i][j] == 0):
        for k in range(1, 10):
            if(isValid(sudoku, i, j, k)):
                sudoku[i][j] = k
                SudokuSolver(sudoku, i, j + 1)
                if(_break == True):
                    return
        sudoku[i][j] = 0
    else: 
        SudokuSolver(sudoku, i, j + 1)

SudokuSolver(sudoku, 0, 0)  
print("\nSolved Sudoku:\n")
for i in range(9):
    print(sudoku[i])

# cv2.imshow("Cropped_gray_cropped_image", bitwise_not_thres_img)
# cv2.waitKey(0)