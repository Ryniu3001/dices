from __future__ import print_function
import os
import numpy as np
from matplotlib import pyplot as plt
import math
import cv, cv2

def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)



def main():
    PLIKI = 6                                  # ktore pliki zaladowac
    plt.figure(figsize=(16, 18))                # dla jednej kostki (16, 18) dla zbioru (50, 60)
    dices = []                                  #
    for file in os.listdir("./kostki/"):        #
     if file.endswith("(%d).jpg" % PLIKI):              #
          dices += ["./kostki/" + file]       # Zbior kostek
    #dices = ["./kostki/10(6).jpg"]             # jedna kostka
    print ("Ilosc zdjec: ",len(dices))
    i = 0
    verify = 0
    for file in dices:
        print ("Zdjecie: ",file)
        ax = plt.subplot(math.ceil(len(dices)/2.0), 2, i)

        oryg = cv2.imread(file)
        grey = cv2.cvtColor(oryg, cv.CV_RGB2GRAY)
        grey = gamma_correction(grey,1.5)
        grey = cv2.fastNlMeansDenoising(grey, None, 11, 7, 21)  #znaczaco wydluza obliczenia
        grey = cv2.medianBlur(grey, 5)
        edges = cv2.Canny(grey,90,120)
        kernel = np.ones((1,1),np.uint8)
        dil = cv2.dilate(edges, kernel,iterations = 2)
        th2 = cv2.morphologyEx(dil, cv2.MORPH_OPEN, kernel)

        repeat = 0

        (cnts, _) = cv2.findContours(th2.copy() , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None


        cnts = cnts[repeat:]

        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            #print (len(approx))
            if(max(approx[:,0,0]) - min(approx[:,0,0]) < 900):
                screenCnt = approx
                break

        #cv2.drawContours(oryg, [screenCnt], -1, (0,255,0), 3)

        a = 200
        h = len(oryg)                #wyoskosc
        w = len(oryg[0])               #szerokosc
        xmax = max(screenCnt[:, 0, 0]) + a
        xmin = min(screenCnt[:, 0, 0]) - a
        ymax = max(screenCnt[:, 0, 1]) + a
        ymin = min(screenCnt[:, 0, 1]) - a

        xmin = 0 if xmin<0 else xmin
        ymin = 0 if ymin<0 else ymin
        ymax = h if ymax>h else ymax
        xmax = w if xmax>w else xmax

        small_image = oryg[ymin:ymax,xmin:xmax]

        if len(small_image) == 0:
            print("Brak rozpoznania")
            continue

        th5 = cv2.blur(th2[ymin:ymax,xmin:xmax],(3,3))
        #th5 = cv2.adaptiveThreshold(th5, 10, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 3)

        max_radius = 60
        circles = cv2.HoughCircles(th5, cv2.cv.CV_HOUGH_GRADIENT, 1, 30, np.array([]), 10, 30, 5, max_radius) # 30 / 27

        while (circles is not None) and (len(circles[0]) > 6):
            max_radius -= 5
            circles = cv2.HoughCircles(th5, cv2.cv.CV_HOUGH_GRADIENT, 1, 30, np.array([]), 10, 30, 5, max_radius)
            if max_radius < 30:
                continue

        if circles is not None:
            for c in circles[0]:
                cv2.circle(small_image, (c[0], c[1]), c[2], (0, 255, 0), 2)
                #edges = cv2.Canny( blur, 40, 80 )
            print(circles[0])
            if len(circles[0]) == int(file[-6]):
                verify += 1

        ax.imshow(small_image)
        #plt.subplot(1,2,i+1), plt.imshow(th5)
        ax.axis('off')
        ax.set_title(file)
        i += 1
    print (str(verify) + " / " + str(len(dices)))
    #plt.savefig('countours%d.pdf' % PLIKI)
    plt.show()

if __name__ == '__main__':
    main()