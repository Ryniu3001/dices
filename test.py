from __future__ import print_function
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import cv, cv2
def main():
    plt.figure(figsize=(60, 90))                # dla jednej kostki (16, 18) dla zbioru (50, 60)
    dices = []                                  #
    for file in os.listdir("./kostki/"):        #
     if file.endswith("(2).jpg"):              #
          dices += ["./kostki/" + file]       # Zbior kostek
    #dices = ["./kostki/46(2).jpg"]             # jedna kostka
    i = 0
    for file in dices:
        ax = plt.subplot(7, 2, i)

        oryg = cv2.imread(file)
        grey = cv2.cvtColor(oryg, cv.CV_RGB2GRAY)
        #grey = cv2.fastNlMeansDenoisingColored(oryg,None,10,10,7,21)
        grey = cv2.fastNlMeansDenoising(grey, None, 12,7,21)  #znaczaco wydluza obliczenia
        grey = cv2.medianBlur(grey, 5)
        edges = cv2.Canny(grey,90,120)
        kernel = np.ones((2,2),np.uint8)
        dil = cv2.dilate(edges, kernel,iterations = 2)
        dil = cv2.morphologyEx(dil, cv2.MORPH_OPEN, kernel)


        circles =  cv2.HoughCircles(dil, cv2.cv.CV_HOUGH_GRADIENT, 2, 30, np.array([]), 30, 45, 5, 35)
        if circles is not None:
            for c in circles[0]:
                cv2.circle(oryg, (c[0],c[1]), c[2], (0,255,0),2)
                #edges = cv2.Canny( blur, 40, 80 )

            print(circles[0])
        ax.imshow(oryg, cmap=plt.get_cmap("gray"))
        #plt.subplot(1,2,i+1), plt.imshow(dil, cmap=plt.get_cmap("gray"))
        ax.axis('off')
        ax.set_title(file)
        i += 1
    plt.savefig('countours2.pdf')
    #plt.show()


if __name__ == '__main__':
    main()

aerfaegsfg
sdfg
sdfgs
dfgsdfgsdfgsdf
gsd
fg
dsfg
