from __future__ import print_function
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import cv, cv2
def main():
    plt.figure(figsize=(16 , 18))                # dla jednej kostki (16, 18) dla zbioru (50, 60)
    dices = []                                  #
    #for file in os.listdir("./kostki/"):        #
     #  if file.endswith("(6).jpg"):              #
    #      dices += ["./kostki/" + file]       # Zbior kostek
    dices = ["./kostki/76(3).jpg"]             # jedna kostka
    print (len(dices))
    i = 0
    verify = 0
    for file in dices:
        ax = plt.subplot(1, 2, i)

        oryg = cv2.imread(file)
        grey = cv2.cvtColor(oryg, cv.CV_RGB2GRAY)
        grey **= 0.5
        print (len(oryg[0][:]))

        #grey = cv2.fastNlMeansDenoising(grey, None, 11, 7, 21)  #znaczaco wydluza obliczenia
        #grey = cv2.medianBlur(grey, 5)
        #grey = cv2.blur(grey,(5,5))


        edges = cv2.Canny(grey,1,200)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        #dil = cv2.dilate(grey, kernel,iterations = 1)
        #dil = cv2.morphologyEx(dil, cv2.MORPH_OPEN, kernel)

        '''
        circles =  cv2.HoughCircles(edges, cv2.cv.CV_HOUGH_GRADIENT, 2, 30, np.array([]), 30, 35, 5, 40)
        if circles is not None:
            for c in circles[0]:
                cv2.circle(oryg, (c[0],c[1]), c[2], (0,255,0),2)
                #edges = cv2.Canny( blur, 40, 80 )
            if len(circles[0]) == 3:
                verify += 1
            print (circles[0])
        '''
        ax.imshow(oryg)
        plt.subplot(1,2,i+1), plt.imshow(edges)
        ax.axis('off')
        ax.set_title(file)
        i += 1
    print (str(verify) + " / " + str(len(dices)))
    #plt.savefig('countours6.pdf')
    #plt.show()


if __name__ == '__main__':
    main()