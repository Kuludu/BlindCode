import cv2
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import precision_score,recall_score

TRAIN_COUNT = 500
TEST_COUNT = 100

def get_features(object_detect, count, test=False):
    if test:
        img_path = f"data/test_set/{object_detect}s/{object_detect}.%d.jpg"
        start = 4001
    else:
        img_path = f"data/training_set/{object_detect}s/{object_detect}.%d.jpg"
        start = 1


    if object_detect == "cat":
        labels = np.array([0 for _ in range(count)]).reshape(-1, 1)
    else:
        labels = np.array([1 for _ in range(count)]).reshape(-1, 1)


    features = list()
    for i in range(start, start+count):
        print(img_path % i)
        # 读取图片
        gray = cv2.imread(img_path % i, cv2.IMREAD_GRAYSCALE)
        # 尺寸缩放
        gray = cv2.resize(gray, (128, 128))
        # 中值滤波
        gray = cv2.medianBlur(gray, 3)
        # HOG特征提取
        hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8))
        features.append(hog_image.flatten())
    features = np.array(features)
    return features, labels


def get_predict_img(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 尺寸缩放
    gray = cv2.resize(gray, (128, 128))
    # 中值滤波
    gray = cv2.medianBlur(gray, 3)
    normalised_blocks, hog_image = hog(gray, orientations=9, pixels_per_cell=( 8, 8), cells_per_block=(8, 8), visualise=True)
    return hog_image.reshape(1, -1)

cat, cat_labels  = get_features(object_detect="cat", count=TRAIN_COUNT)
dog, dog_labels = get_features(object_detect="dog", count=TRAIN_COUNT)
img = np.vstack([cat, dog])
labels = np.vstack([cat_labels, dog_labels])
res = np.hstack([img, labels])

clf = SVC(probability=True)
data = res[:, :-1]
labels = res[:, -1]
clf.fit(data, labels)


# ----------- 预测单张图片 ---------------------------------
# test_img = get_predict_img("training_set/cats/cat.38.jpg")
# pred = clf.predict(test_img)
# print(pred)
# ----------- 预测单张图片 ---------------------------------

test_cat, test_cat_labels = get_features(object_detect="cat", count=TEST_COUNT, test=True)
test_dog, test_dog_labels = get_features(object_detect="dog", count=TEST_COUNT, test=True)

test_img = np.vstack([test_cat, test_dog])
test_labels = np.vstack([test_cat_labels, test_dog_labels])

pred = clf.predict(test_img)

precision = precision_score(pred,test_labels)
recall = recall_score(pred,test_labels)
print("实际类别:",test_labels.flatten())
print("预测类别:",pred.flatten())
print(f"精准率:{precision}, 召回率:{recall}")
