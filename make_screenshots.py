import ctypes
from datetime import datetime
import time
import cv2  # instal as opencv-python
import numpy as np
import pyautogui.screenshotUtil
from pynput.keyboard import Key, Listener

#dt = datetime.microsecond  # Get timezone naive now


def run_script():
    looperCPU = 20
    start = time.time()
    while (looperCPU != 0):
        time.sleep(1)
        makeScreen()
        looperCPU -= 1



def makeScreen():
    #seconds = dt.timestamp()
    seconds = time.time()

    print(seconds)
    image = pyautogui.screenshot()
    image2 = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
   # print( cv2.imwrite.__doc__)
    cv2.imwrite("Medkit_Other/"+seconds.__str__() + ".png", image2)




def run():

    ctypes.windll.user32.MessageBoxW(0, "SCP IMAGE is now active", "Start", 0x1000)
    print("Start")

    def on_release(key):

        if key == Key.alt_r:

                    run_script()
                    print("End")

        if key == Key.scroll_lock:
            ctypes.windll.user32.MessageBoxW(0, "Script stopped", "Exiting", 0x1000)
            listener.stop()

    with Listener(
            on_press=None,
            on_release=on_release) as listener:
        listener.join()


run()



"""while (looperCPU != 0):
    start_time = time.time()
    # Do some stuff
    while printerLooper == True :
        print("Sleeping for ", secondsPause, " seconds")
        print(random_number)
        printerLooper = False
    end_time = time.time()

    print("total time taken this loop: ", end_time - start_time)"""