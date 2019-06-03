import matplotlib.pyplot as plt
import os


def makeTable_img(img, mytitle, i, quantity):
    q_2 = quantity // 3
    cols = 3
    if quantity % 2 == 1:
        rows = q_2 + 1
    else:
        rows = q_2
    if i > rows:
        i = i - rows
    drawTable(rows, cols, img, i, mytitle)


def drawTable(rows, cols, img, i, mytitle):
        plt.subplot( cols,rows, i), plt.imshow(img, cmap='gray')
        plt.rcParams.update({'font.size': 7})
        plt.title(mytitle)  , plt.xticks([]), plt.yticks([])


# def makeTable_arrImg(ArrImg, ArrTitle):
#     if len(ArrImg) == len(ArrTitle):
#         try:
#             print(title)
#             plt.imshow(img, cmap="gray")
#             plt.show()
#         except Exception as e:
#             print(e)
#             pass
#     else:
#         print("Length of array of images and titles are not equal!")
