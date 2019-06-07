import cv2  # instal as opencv-python
import ctypes
from pynput.keyboard import Key, KeyCode, Listener# import time
import os
import numpy as np
import tensorflow as tf
import pyautogui.screenshotUtil

DATADIR = "C:\\Users\\Ira\PycharmProjects\Scp_medkits"
IMG_SIZE = 60 #depends on dataset
inv_coordArr = [(1080, 325), (1250, 450), (1250, 720), (850, 900), (1080, 900), (660, 730), (660, 470), (830, 300)]
model = tf.keras.models.load_model("3 - CNV - 64 - ND - 0 - DNS - 1559894573.model",compile=True)

# global graph, model
# graph = tf.get_default_graph()


# def prepare(img):
#     new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def m_dft(orig_img):
    img = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT) #discrete Fourier transform (DFT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])).astype('uint8')
    return magnitude_spectrum.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def run_predict(img_):
    try:
        if img_.shape:
                prediction = model.predict([m_dft(img_)])
                if prediction == 1:
                    category = 1
                    # c = "Medkit"
                else:
                    category = 0
                    # c = "Not Medkit"

                # print(c)
                # plt.imshow(img_, cmap="gray")
                # plt.show()
                return category
        else:
            print("Dos not exist")

    except Exception as e:
        print(e)
        pass



def select(mat, point0, point1):
    xmax = max(point0[0], point1[0])
    xmin = min(point0[0], point1[0])

    ymax = max(point0[1], point1[1])
    ymin = min(point0[1], point1[1])

    return mat[ymin:ymax, xmin:xmax]


def my_crop(img_mask, img):
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)  # ХРОБИ ТАК ШОБ ЗНАХОДИЛОКВАДРАТ
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    [[x1, y1], [x0, y1], [x0, y0], [x1, y0]] = np.int0(box)
    return select(img, [x0, y0], [x1, y1])


def select_mask(img):
    path_mask = os.path.join(DATADIR, "MSC_SQAURE")
    scaled_v_img = img[180:1030, 530:1385]

    for indx, img_m in enumerate(os.listdir(path_mask)):
        try:
            img_mask = cv2.imread(os.path.join(path_mask, img_m), cv2.IMREAD_GRAYSCALE)
            selected_img = cv2.bitwise_and(scaled_v_img, scaled_v_img, mask=img_mask)
            res = my_crop(img_mask, selected_img)

            height_к, width_к = res.shape[:2]
            if height_к >= IMG_SIZE or width_к >= IMG_SIZE:
                scaled_res = cv2.resize(res, (IMG_SIZE, IMG_SIZE))

                predict_res = run_predict(scaled_res)
                if predict_res == 1:
                    return indx
        except Exception as e:
            print(e)
            pass
    return -1


def click_inv(index_inv):
    x, y = inv_coordArr[index_inv]
    pyautogui.click(x, y)
    pyautogui.PAUSE = 0.3
    pyautogui.click(x, y)
    pyautogui.press('alt')


def run_medkit():
    scrn_g = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_BGR2GRAY)
    index_inv = select_mask(scrn_g)
    print('Index: ',index_inv)
    if index_inv == -1:
        print("No medkit")
    else:
        click_inv(index_inv)


def run():
    ctypes.windll.user32.MessageBoxW(0, "Scp medkits runs now", "Start", 0x1000)

    def on_release(key):

        if key == KeyCode.from_char('f'):
            pyautogui.press('alt')
            run_medkit()

        if key == Key.scroll_lock:
            ctypes.windll.user32.MessageBoxW(0, "Script stopped", "Exiting", 0x1000)
            return False

    # with Listener(
    #         on_press=None,
    #         on_release=on_release) as listener:
    #     listener.join()

    with Listener(
            on_press=None,
            on_release=on_release,
            suppress=False) as listener:
        try:
            listener.join(0)
        except Exception as e:
            print(e)

    # listener = Listener(
    #     on_press=None,
    #     on_release=on_release,
    #     suppress=False)
    # listener.start()


run()
