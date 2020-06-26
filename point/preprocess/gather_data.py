import os
import cv2
import numpy as np
def binaryMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new
def checker(image,x,y):
    for i in range(4):
        if x+i<300:
            if image[x+i][y]!=0:
                return False
        else :
            return False
    for i in range(4):
        if y+i<300:
            if image[x][y+i]!=0:
                return False
        else :
            return False
    return True

def saveimg(image):
    destn = "../dataset/dummy/"
    image = binaryMask(image)
    posx=0
    posy=0
    flag=0
    for i in range(300):
        for j in range(300):
            if image[i][j]== 0:
                if checker(image,i,j):
                        flag=1
                        posx=i
                        posy=j
                        break
        if flag==1:
            image = cv2.resize(image, (300,300))
            cv2.imshow("Saving Image",image)
            cv2.imwrite(destn+str(posx)+","+str(posy)+".png",image)
            break
rec = cv2.VideoCapture(0)
while True:
    _,frame = rec.read()
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.rectangle(frame,(0,80),(330,420),(0,0,0),2)
    cv2.imshow("VideoInput",frame)
    if key == ord('s'):
        hand = frame[100:400, 10:310]
        saveimg(hand)
