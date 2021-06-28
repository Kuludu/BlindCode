from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from skimage.feature import hog
from skimage import io
import pickle
import qdarkstyle


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = loadUi('UI/main.ui', self)
        self.ui.setWindowTitle("CV-HogSVM")

        self.ui.load_image.clicked.connect(self.load_image_func)
        self.ui.resize_button.clicked.connect(self.resize_func)
        self.ui.image_gray.clicked.connect(self.gray_func)
        self.ui.median_button.clicked.connect(self.median_func)
        self.ui.features_button.clicked.connect(self.get_features_func)
        self.ui.load_model.clicked.connect(self.load_model_func)
        self.ui.model_predict.clicked.connect(self.model_predict_func)
        self.ui.handle_image.clicked.connect(self.handle_image_func)

        self.ui.image_path.setText("未选择文件")

        self.handle_image = None
        self.show_image("img/dog_and_cat.jpeg")

    def handle_image_func(self):
        self.gray_func()
        self.resize_func()
        self.median_func()
        self.get_features_func()

    def show_image(self,image_path):
        lbl = self.ui.image_label
        pixmap = QPixmap(image_path) 
        lbl.setPixmap(pixmap)
        lbl.setScaledContents(True)


    def load_image_func(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName( self.ui, "请选择你要打开图片的名字", "./data/test_set/", "Image files(*.jpg)")[0]
        image_path = file_name[file_name.find("data"):]
        self.ui.image_path.setText(image_path)
        self.show_image(image_path)
        self.handle_image = cv2.imread(image_path)

    def gray_func(self):
        self.handle_gray_image = cv2.cvtColor(self.handle_image, cv2.COLOR_BGR2GRAY)
        path = ".cache/gray_image.jpg"
        cv2.imwrite(path,self.handle_gray_image)
        self.show_image(path)


    def resize_func(self):
        gray = self.handle_gray_image
        self.handle_resize_image = cv2.resize(gray, (128, 128))
        path = ".cache/resize_image.jpg"
        cv2.imwrite(path,self.handle_resize_image)
        self.show_image(path)

    def median_func(self):
        resize = self.handle_resize_image
        self.handle_median = cv2.medianBlur(resize, 3)
        path = ".cache/median_image.jpg"
        cv2.imwrite(path,self.handle_median)
        self.show_image(path)

    def get_features_func(self):
        median_resize = self.handle_median
        self.normalised_blocks, self.features_image = hog(median_resize, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),visualize=True)
        path = ".cache/features_image.jpg"
        io.imsave(path, self.features_image)
        self.show_image(path)
        
    def load_model_func(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self.ui, "请选择你要加载模型的名字", "model", "Model files(*.pkl)")[0]

        with open(file_name,'rb') as f:
            self.model = pickle.load(f)

    def model_predict_func(self):
        predict = self.normalised_blocks.reshape(1,-1)
        labels = ["猫","狗"]
        y = int(self.model.predict(predict)[0])
        s1 = str(labels[y])
        self.ui.result.setText(s1)


app = QApplication([])
windows = MainWindow()
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
windows.setWindowTitle('CV-HogSVM')
windows.show()
app.exec_()
