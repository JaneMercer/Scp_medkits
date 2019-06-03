import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from my_lib import makeTable_img

CATEGORIES = ["MEDKIT", "Not"]

DATADIR = "C:\\Users\\Ira\PycharmProjects\Scp_medkits\\test_img"


def prepare(img):
    IMG_SIZE = 60
    new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



# model = tf.keras.models.load_model("3 - CNV - 64 - ND - 0 - DNS - 1559109706.model")
#
# for img in os.listdir(DATADIR):
#     i = os.listdir(DATADIR).index(img)+1
#     try:
#         my_str = os.path.join(DATADIR, img)
#         img_ = cv2.imread(my_str, cv2.IMREAD_GRAYSCALE)
#         if img_.shape:
#             prediction = model.predict([prepare(img_)])
#             if prediction == 1:
#                     category = "Medkit"
#             else:
#                     category = "Not Medkit"
#            # print(category)
#             makeTable_img(img_, category, i, len(os.listdir(DATADIR)))
#         else:
#             print("Dos not exist")
#
#     except Exception as e:
#         print(e)
#         pass
# plt.show()
# prediction = model.predict([prepare('med-test.jpg')])
# prediction = model.predict([prepare('3325.png')])
# print(prediction)
# prediction = model.predict([prepare('res1.png')])
# print(prediction)
# prediction = model.predict([prepare('1553811702.729782.png')])
# print(prediction)


def run_predict(img_):
    model = tf.keras.models.load_model("3 - CNV - 64 - ND - 0 - DNS - 1559109706.model")
    results = []

    try:
        if img_.shape:
                prediction = model.predict([prepare(img_)])
                if prediction == 1:
                    category = "Medkit"
                else:
                    category = "Not Medkit"
                print(category)
               # makeTable_img(img_, category, i, len(os.listdir(DATADIR)))
        else:
                print("Dos not exist")

    except Exception as e:
            print(e)
            pass
    #plt.show()

