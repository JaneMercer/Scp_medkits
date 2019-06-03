import cv2  # instal as opencv-python
#import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import pyautogui.screenshotUtil

DATADIR = "C:\\Users\\Ira\PycharmProjects\Scp_medkits"
IMG_SIZE = 60
model = tf.keras.models.load_model("3 - CNV - 64 - ND - 0 - DNS - 1559109706.model")

def prepare(img):
    IMG_SIZE = 60
    new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def run_predict(img_):

    try:
        if img_.shape:
                prediction = model.predict([prepare(img_)])
                if prediction == 1:
                    category = "Medkit"
                else:
                    category = "Not Medkit"

                print(category)
                plt.imshow(img_, cmap="gray")
                plt.show()
        else:
                print("Dos not exist")

    except Exception as e:
            print(e)
            pass
    return category


def select(mat, point0, point1):
    xmax = max(point0[0], point1[0])
    xmin = min(point0[0], point1[0])

    ymax = max(point0[1], point1[1])
    ymin = min(point0[1], point1[1])

    return mat[ymin:ymax, xmin:xmax]


def my_crop(img_mask, img):
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # ХРОБИ ТАК ШОБ ЗНАХОДИЛОКВАДРАТ
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    [[x1, y1], [x0, y1], [x0, y0], [x1, y0]] = np.int0(box)
    return select(img, [x0, y0], [x1, y1])


def select_mask(img):
    path_mask = os.path.join(DATADIR, "MSC_SQAURE")
    img_indx = []
    try:
        scaled_v_img = img[180:1030, 530:1385]
        for img_m in os.listdir(path_mask):
            try:
                    img_mask = cv2.imread(os.path.join(path_mask, img_m), cv2.IMREAD_GRAYSCALE)
                    selected_img = cv2.bitwise_and(scaled_v_img, scaled_v_img, mask=img_mask)
                    res = my_crop(img_mask, selected_img)

                    height_к, width_к = res.shape[:2]
                    if height_к < 50 or width_к < 50:
                        plt.imshow(res, cmap="gray")
                        plt.show()
                    else:
                        scaled_res = cv2.resize(res, (IMG_SIZE, IMG_SIZE))

                        predict_res = run_predict(scaled_res) #add import p-Predict
                        #img_indx.append((img_m, predict_res))
                        if predict_res == 1:
                            return img_m
            except Exception as e:
                    print(e)
                    pass

    except Exception as e:
        print(e)
        pass


scrn = cv2.imread('for-testing.png', 0)

select_mask(scrn)


#scrn = pyautogui.screenshot()
