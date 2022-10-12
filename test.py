from nd2reader import ND2Reader
import cv2 as cv
import numpy as np
import time

#t1 = time.time()

nd2_file = ND2Reader('./../DataForTracking/Large mvt 1.nd2')

# c=4 : "normal" image
# c=0 : cell image
"""
for i in range(8):
    img = nd2_file.get_frame_2D(x=0, y=0, c=4, t=i, z=0, v=0)
    cv.imwrite(f"images/img{i}.png", img[:2000,:2000])
"""

for i in range(8):
    img = nd2_file.get_frame_2D(x=0, y=0, c=4, t=i, z=0, v=0)[1000:2000,1000:2000]
    img_cells = nd2_file.get_frame_2D(x=0, y=0, c=0, t=i, z=0, v=0)[1000:2000,1000:2000]

    # type conversion for cv2
    img_cells = cv.normalize(src=img_cells, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    thresh = 20
    im_bw = cv.threshold(img_cells, thresh, 255, cv.THRESH_BINARY)[1]

    cells = cv.bitwise_and(img_cells, img_cells, mask = im_bw)


    mask = cv.blur(im_bw, (30,30))

    img = cv.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, dp=1, param1=50, param2=30, minDist=40, minRadius=20, maxRadius=25)

    circles = np.uint16(np.around(circles))
    for c in circles[0,:]:
        # draw the outer circle
        cv.circle(cimg,(c[0],c[1]),c[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(cimg,(c[0],c[1]),2,(0,0,255),3)

    cimg[im_bw == 255] = [255,0,0]
    cimg = cv.bitwise_and(cimg, cimg, mask = mask)
    cv.imwrite(f"images/hough_mask_img{i}.png", cimg)

#print(time.time() - t1)

cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()