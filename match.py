import cv2
import numpy as np
from matplotlib import pyplot as plt



img_gray = cv2.imread('1_1.png',0)

#print(plt.imshow.__doc__)
#img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('1.png',0)
y_1, x_1 = template.shape[::-1]
y_0, x_0 = img_gray.shape

hog = cv2.HOGDescriptor()
#hog = cv2.HOGDescriptor("hog.xml")
winStride = (8, 8)
padding = (2, 2)
descriptor_temp = hog.compute(template, winStride, padding)
#hog.save("hog.xml")
print(hog.detectMultiScale.__doc__)

#descriptor_orig2 = hog.compute(img_gray)
hog2 = cv2.HOGDescriptor()
descriptor_orig = hog2.compute(img_gray)
if descriptor_orig is not None: descriptor_orig = descriptor_orig.ravel()

(rects, weights) = hog.detectMultiScale(template, winStride=(8, 8),padding=(2, 2), scale=1.05)


for (x, y, w, h) in rects:
    cv2.rectangle(img_gray, (x, y), (x + w, y + h), 255, 2)
  #  rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
  #  pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 #   for (xA, yA, xB, yB) in rects:
   #    cv2.rectangle(img_gray, (xA, yA), (xB, yB), (0, 255, 0), 2)
    #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)





print(descriptor_orig)
#plt.imshow(h, 'gray')
#plt.show()


#print(x_0, ">", x_1, " & ", y_0, ">", y_1 )
"""
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

threshold = 0.7
loc = np.where( res >= threshold)

print(loc)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_gray, pt, (pt[0] + y_1, pt[1] + x_1), (0,0,255), 2)
    cv2.imwrite('res2.png',img_gray)


if x_0 > x_1 and y_0 > y_1:
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_gray, pt, (pt[0] + y_1, pt[1] + x_1), (0,0,255), 2)
        cv2.imwrite('res1.png',img_gray)
else:
    print("error")
"""