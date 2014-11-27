from __future__ import print_function
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from timeit import default_timer as timer
import cv, cv2
def main():
    plt.figure(figsize=(16, 18))                # dla jednej kostki (16, 18) dla zbioru (50, 60)
    dices = []                                  #
    #for file in os.listdir("./kostki/"):        #
    #    if file.endswith("(6).jpg"):              #
    #        dices += ["./kostki/" + file]       # Zbior kostek
    dices = ["./kostki/62(6).jpg"]             # jedna kostka
    print (len(dices))
    i = 0
    verify = 0
    for file in dices:

        ax = plt.subplot(1, 2, i)
        ory = cv2.imread(file)
        oryg = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        grey = cv2.blur(oryg,(3,3))
        #grey = cv2.bilateralFilter(oryg, 11, 17, 17)

        th2 = cv2.fastNlMeansDenoising(grey, None, 30, 7, 21)  #znaczaco wydluza obliczenia 40
        #th2 = cv2.Canny(grey,50,100)
        th2 = cv2.adaptiveThreshold(th2, 10, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, 5) #29 5

        #kernel = np.ones((4,4), np.uint8)
        #edges = cv2.dilate(edges, kernel,iterations = 1)
        #edges = cv2.Canny(th2,40,60)
        #edges = cv2.dilate(edges, kernel,iterations = 1)
        kernel = np.ones((3,3),np.uint8)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        th2 = cv2.dilate(th2, kernel,iterations = 1)
        #th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)


        (cnts, _) = cv2.findContours(th2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None
        cnts = cnts[1:]

        # loop over our contours

        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            #print (len(approx))
            if len(approx) == 4:
                screenCnt = approx
                break

        cv2.drawContours(ory, [screenCnt], -1, (0,255,0), 3)
        print (screenCnt)
        rect = cv2.rectangle(ory,(1611,792),(1443,644), color=200)
        blank_image = np.zeros((len(ory)+148,len(ory[0])+168,3), np.uint8)
        cv2.copyMakeBorder(ory, blank_image, 1,1,1,1 )

        '''
        circles =  cv2.HoughCircles(th2, cv2.cv.CV_HOUGH_GRADIENT, 2, 30, np.array([]), 10, 40, 10, 40)
        if circles is not None:
            for c in circles[0]:
                cv2.circle(ory, (c[0],c[1]), c[2], (0,255,0),2)
                #edges = cv2.Canny( blur, 40, 80 )
            if len(circles[0]) == 1:
                verify += 1
            print (circles[0])
        '''
        ax.imshow(ory)
        plt.subplot(1,2,i+1), plt.imshow(th2)
        ax.axis('off')
        ax.set_title(file)
        i += 1

    print (str(verify) + " / " + str(len(dices)))
    #plt.savefig('countours6.pdf')
    plt.show()


if __name__ == '__main__':
    main()