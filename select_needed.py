import cv2  # instal as opencv-python
import time
import os
import random
import numpy as np
import pickle

DATADIR = "C:\\Users\\Ira\PycharmProjects\Scp_templateMatch"
CATEGORIES_RD = ["SCR_EVRTHNG","SCR_MEDKIT I medkit"]
CATEGORIES_WR = ["TRN_DATA_Not", "TRN_DATA_MEDKIT"]

training_data = []
IMG_SIZE = 60
X = []
Y = []


# scaled_mask_img = img_mask[180:1030, 530:1385]
# cv2.imwrite('mask.png', scaled_mask_img)
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

def create_training_data():
    for category in CATEGORIES_RD:
        class_num = CATEGORIES_RD.index(category)
        img_mask_main = cv2.imread('mask-main.png', 0)
        path = os.path.join(DATADIR, category)
        path_wr = os.path.join(DATADIR, CATEGORIES_WR[class_num])
        path_mask = os.path.join(DATADIR, "MSC_SQAURE")

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                scaled_v_img = img_array[180:1030, 530:1385]
                height_s, width_s = scaled_v_img.shape[:2]
                height_m, width_m = img_mask_main.shape[:2]
                if height_s == height_m and width_s == width_m:
                   # imask = cv2.bitwise_and(scaled_v_img, scaled_v_img, mask=img_mask_main)
                    #img_array_blured[imask] = img_array[imask]
                    # plt.imshow(imask, cmap="gray")
                    # plt.show()
                    for img_m in os.listdir(path_mask):
                        try:
                            img_mask = cv2.imread(os.path.join(path_mask, img_m), cv2.IMREAD_GRAYSCALE)
                            selected_img = cv2.bitwise_and(scaled_v_img, scaled_v_img, mask=img_mask)
                            res = my_crop(img_mask, selected_img)
                            if res.size:
                                scaled_res = cv2.resize(res, (IMG_SIZE, IMG_SIZE))
                                # plt.imshow(scaled_res, cmap="gray")
                                # plt.show()
                                str_wr = time.time().__str__() + ".png"
                                cv2.imwrite(os.path.join(path_wr, str_wr), res)
                                training_data.append([scaled_res, class_num])
                            else:
                                print("fail")
                        except Exception as e:
                            print(e)
                            pass

            except Exception as e:
                print(e)
                pass

            # plt.imshow(scaled_v_img, cmap="gray")
            # plt.show()


create_training_data()
print(len(training_data))
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[0])

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # change 1 for 3  for 3 chanels

pickle_out = open("IMGes.pickle", "wb")  # features
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out2 = open("LBLes.pickle", "wb")  # lables
pickle.dump(Y, pickle_out2)
pickle_out2.close()
