from __future__ import print_function
import os
from skimage import data, filter, morphology,color
from skimage.transform import hough_ellipse
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import disk
from scipy import ndimage
from skimage import measure, exposure
from skimage.morphology import square
import cv2
def main():
    plt.figure(figsize=(15, 18))
    #dices = []
    #for file in os.listdir("./kostki/"):
    #  if file.endswith(".jpg"):
    #        dices += ["./kostki/" + file]
    #dices = ["./kostki/13.jpg", "./kostki/7.jpg", "./kostki/1.jpg", "./kostki/4.jpg", "./kostki/3.jpg"]
    dices = ["./kostki/7.jpg"]
    templ = data.imread("./template.jpg",as_grey=True)
    i = 0
    tmpl = data.imread("./template.jpg", as_grey=True)
    for file in dices:
        ax = plt.subplot(1, 1, i)
        sam1 = data.imread(file,  as_grey=True)
        oryg = data.imread(file,  as_grey=False)
        sam2 = exposure.adjust_gamma(sam1,gamma=0.5,gain=1)
        sam2 = filter.canny(sam2, sigma=3)
        sam2 = morphology.dilation(sam2, disk(2))
        #sam2 = ndimage.binary_fill_holes(sam2)

        sam2 = morphology.remove_small_objects(sam2, 500)
        sam2 = morphology.erosion(sam2, square(3))
        sam2 = ndimage.binary_fill_holes(sam2)

        #contours = measure.find_contours(sam2, 0.5)
        #for n, contour in enumerate(contours):
        #ax.plot(contour[:, 1], contour[:, 0], linewidth=2.5)
        ax.imshow(sam2, cmap=plt.get_cmap('gray'))
        ax.axis('off')
        ax.set_title(file)
        i += 1
    #plt.savefig('countours.pdf')
    plt.show()


if __name__ == '__main__':
    main()