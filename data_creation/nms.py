import numpy as np
import cv2 as cv

# Nonmaxima suppression. Finds single points that stick out compared to their 8-neighborhood
def nms (img):
    aux = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype = np.float32)
    aux[1: 1 + img.shape[0], 1: 1 + img.shape[1]] = img
    ans = np.ones(img.shape, dtype = np.float32)
    offx = 1
    offy = 0
    ans = ans * (img > aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])
    offx = -1
    offy = 0
    ans = ans * (img >= aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])
    offx = 0
    offy = 1
    ans = ans * (img > aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])
    offx = 0
    offy = -1
    ans = ans * (img >= aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])

    offx = 1
    offy = 1
    ans = ans * (img > aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])
    offx = -1
    offy = 1
    ans = ans * (img > aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])
    offx = 1
    offy = -1
    ans = ans * (img >= aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])
    offx = -1
    offy = -1
    ans = ans * (img >= aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])

    return ans

# Nonmaxima suppression. Finds single points that stick out compared to their 4-neighborhood
def nms4 (img):
    aux = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype = np.float32)
    aux[1: 1 + img.shape[0], 1: 1 + img.shape[1]] = img
    ans = np.ones(img.shape, dtype = np.float32)
    offx = 1
    offy = 0
    ans = ans * (img > aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])
    offx = -1
    offy = 0
    ans = ans * (img >= aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])
    offx = 0
    offy = 1
    ans = ans * (img > aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])
    offx = 0
    offy = -1
    ans = ans * (img >= aux[1 + offx: img.shape[0] + 1 + offx, 1 + offy: img.shape[1] + 1 + offy])

    return ans


# Nonmaxima suppression canny-style. Finds lines that stick out in the image. Used for example for preprocessing the BF images before hough transform
def canny_nms (img):
    s = img.shape
    aux = np.zeros((s[0] + 2, s[1] + 2), dtype = np.float32)
    aux[1: 1 + s[0], 1: 1 + s[1]] = img

    kernel1 = cv.getGaborKernel((5, 5), 4, 0, 4, 0.5, psi = 0)
    kernel1 = kernel1 - np.mean(kernel1)
    kernel1 = kernel1 / np.max(kernel1)

    kernel2 = cv.getGaborKernel((5, 5), 4, np.pi / 2, 4, 0.5, psi = 0)
    kernel2 = kernel2 - np.mean(kernel2)
    kernel2 = kernel2 / np.max(kernel2)

    kernel3 = cv.getGaborKernel((5, 5), 4, np.pi / 4, 4, 0.5, psi = 0)
    kernel3 = kernel3 - np.mean(kernel3)
    kernel3 = kernel3 / np.max(kernel3)

    kernel4 = cv.getGaborKernel((5, 5), 4, np.pi / 4 + np.pi / 2, 4, 0.5, psi = 0)
    kernel4 = kernel4 - np.mean(kernel4)
    kernel4 = kernel4 / np.max(kernel4)

    # print(kernel1)
    # print(kernel2)
    # print(kernel3)
    # print(kernel4)

    filtered1 = cv.filter2D(img, -1, kernel1, borderType = cv.BORDER_DEFAULT)
    filtered2 = cv.filter2D(img, -1, kernel2, borderType = cv.BORDER_DEFAULT)
    filtered3 = cv.filter2D(img, -1, kernel3, borderType = cv.BORDER_DEFAULT)
    filtered4 = cv.filter2D(img, -1, kernel4, borderType = cv.BORDER_DEFAULT)

    # cv.imshow("test", filtered1)
    # cv.waitKey(0)
    # cv.imshow("test", filtered2)
    # cv.waitKey(0)
    # cv.imshow("test", filtered3)
    # cv.waitKey(0)
    # cv.imshow("test", filtered4)
    # cv.waitKey(0)

    compound = np.asarray([filtered1, filtered2, filtered3, filtered4])

    orientations = np.argmax(compound, axis = 0)
    orientations = np.transpose(np.asarray([orientations == 0, orientations == 1, orientations == 2, orientations == 3], dtype = np.float32), [1, 2, 0])

    domi1 = np.ones(s, dtype = np.float32)
    offx = 0
    offy = 1
    domi1 = domi1 * (img > aux[1 + offx: s[0] + 1 + offx, 1 + offy: s[1] + 1 + offy])
    offx = 0
    offy = -1
    domi1 = domi1 * (img > aux[1 + offx: s[0] + 1 + offx, 1 + offy: s[1] + 1 + offy])


    domi2 = np.ones(s, dtype = np.float32)
    offx = 1
    offy = 0
    domi2 = domi2 * (img > aux[1 + offx: s[0] + 1 + offx, 1 + offy: s[1] + 1 + offy])
    offx = -1
    offy = 0
    domi2 = domi2 * (img > aux[1 + offx: s[0] + 1 + offx, 1 + offy: s[1] + 1 + offy])


    domi3 = np.ones(s, dtype = np.float32)
    offx = 1
    offy = 1
    domi3 = domi3 * (img > aux[1 + offx: s[0] + 1 + offx, 1 + offy: s[1] + 1 + offy])
    offx = -1
    offy = -1
    domi3 = domi3 * (img > aux[1 + offx: s[0] + 1 + offx, 1 + offy: s[1] + 1 + offy])


    domi4 = np.ones(s, dtype = np.float32)
    offx = 1
    offy = -1
    domi4 = domi4 * (img > aux[1 + offx: s[0] + 1 + offx, 1 + offy: s[1] + 1 + offy])
    offx = -1
    offy = 1
    domi4 = domi4 * (img > aux[1 + offx: s[0] + 1 + offx, 1 + offy: s[1] + 1 + offy])

    compound_dominants = np.transpose(np.asarray([domi1, domi2, domi3, domi4]), [1, 2, 0])


    ans = np.max(compound_dominants * orientations, axis = 2)
    return ans


def grad_nms (img):

    grad_x = cv.Scharr(img, -1, 1, 0)
    grad_y = cv.Scharr(img, -1, 0, 1)
    # grad_x = cv.Sobel(img, -1, 1, 0, ksize = 1)
    # grad_y = cv.Sobel(img, -1, 0, 1, ksize = 1)

    # cv.imshow('test',grad_x)
    # cv.waitKey(0)
    # cv.imshow('test',grad_y)
    # cv.waitKey(0)

    # grad_xx = cv.Sobel(img, -1, 2, 0, ksize = 1)
    # grad_xy = cv.Sobel(img, -1, 1, 1, ksize = 1)
    # grad_yy = cv.Sobel(img, -1, 0, 2, ksize = 1)
    grad_xx = cv.Scharr(grad_x, -1, 1, 0)
    grad_yy = cv.Scharr(grad_y, -1, 0, 1)
    grad_xy_sq = cv.Scharr(grad_y, -1, 1, 0) * cv.Scharr(grad_x, -1, 0, 1)

    dets = grad_xx * grad_yy - grad_xy_sq
    ev1 = (grad_xx + grad_yy) * 0.5 + np.sqrt(((grad_xx + grad_yy) * 0.5)**2 - dets)
    ev2 = (grad_xx + grad_yy) * 0.5 - np.sqrt(((grad_xx + grad_yy) * 0.5)**2 - dets)
    # ev1 = grad_xx
    # ev2 = grad_yy
    # cv.imshow('test',(ev1 < 0) * 1.0)
    # cv.waitKey(0)
    # cv.imshow('test',(ev2 < 0) * 1.0)
    # cv.waitKey(0)
    # cv.imshow('test',(ev1 < 0) * (ev2 < 0) * 1.0)
    # cv.waitKey(0)

    grad_norm = np.linalg.norm(np.asarray([grad_x, grad_y]), axis = 0)
    # cv.imshow('test',grad_norm / grad_norm.max())
    # cv.waitKey(0)
    gradient_minima = nms(-grad_norm)
    # cv.imshow('test',gradient_minima)
    # cv.waitKey(0)
    # cv.imshow('test',(ev1 < 0) * (ev2 < 0) * 1.0)
    # cv.waitKey(0)
    # cv.imshow('test',gradient_minima)
    # cv.waitKey(0)
    # cv.imshow('test',(ev1 < 0) * (ev2 < 0) * 1.0)
    # cv.waitKey(0)
    # cv.imshow('test',gradient_minima)
    # cv.waitKey(0)
    # cv.imshow('test',(ev1 < 0) * (ev2 < 0) * 1.0)
    # cv.waitKey(0)
    nd_ness = np.logical_and(ev1 < 0, ev2 < 0) * 1.0
    gradient_minima = gradient_minima * nd_ness
    # cv.imshow('test',gradient_minima)
    # cv.waitKey(0)

    return gradient_minima