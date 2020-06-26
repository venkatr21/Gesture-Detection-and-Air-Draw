import os
import cv2
import numpy as np
def checker(image,x,y):
    for i in range(8):
        if list(image[x+i][y])!=[0,0,0]:
            return False
    for i in range(8):
        if list(image[x][y+i])!=[0,0,0]:
            return False
    return True
    
    
filename = "../../gesture/dataset/train/1/"
destn = "../dataset/train/"
for root,dirs,files in os.walk(filename):
    for imgname in files:
        posx=0
        posy=0
        flag=0
        image = cv2.imread(filename+imgname)
        for i in range(300):
            for j in range(300):
                if list(image[i][j])== [0,0,0]:
                    if checker(image,i,j):
                        flag=1
                        posx=i
                        posy=j
                        break
            if flag==1:
                cv2.imwrite(destn+str(posx)+","+str(posy)+".png",image)
                break
