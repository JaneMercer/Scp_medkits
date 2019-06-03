import ctypes
import tkinter as tk

import cv2  # instal as opencv-python
import numpy as np
import pyautogui.screenshotUtil
from pynput.keyboard import Key, Listener



def run_script(mouse_position_1):
    r = tk.Tk()
    r.withdraw()
    if not r.cget('cursor') == 'wait':
        r.config(cursor='wait')
        print(r.cget('cursor'))
    r.update()
    r.clipboard_clear()
    text_result = screen_to_text(mouse_position_0, mouse_position_1)
    r.clipboard_append(text_result)
    r.update()
    if not r.cget('cursor') == "":
        r.config(cursor="")


def screen_to_imgArr():
    image = pyautogui.screenshot()
    image2 = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    imgArr = select_mask(image)
    return imgArr


def run():

    ctypes.windll.user32.MessageBoxW(0, "Script is now active", "Start", 0x1000)
    print("Start")
    global mouse_position_0
    mouse_position_0 = (0, 0)

    def on_release(key):

        if key == Key.alt_r:
            global mouse_position_0
            if mouse_position_0 == (0, 0):
                mouse_position_0 = pyautogui.position()
            else:

                mouse_position_1 = pyautogui.position()
                if pyautogui.onScreen(mouse_position_0) and pyautogui.onScreen(mouse_position_1):

                    run_script(mouse_position_1)
                    mouse_position_0 = (0, 0)
                    print("End")
                else:
                    print("ERROR: Mouse is not on screen")

        if key == Key.scroll_lock:
            ctypes.windll.user32.MessageBoxW(0, "Script stopped", "Exiting", 0x1000)
            listener.stop()

    with Listener(
            on_press=None,
            on_release=on_release) as listener:
        listener.join()


run()
