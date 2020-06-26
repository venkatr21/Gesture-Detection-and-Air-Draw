import cv2
import tensorflow as tf
import numpy as np

def binaryMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new
def queuepush(pt,x,y):
    if len(pt)<20:
        pt.append((y,x))
    else :
        pt.pop(0)
        pt.append((y,x))
    return pt
classes = 'ZERO ONE TWO THREE FOUR FIVE'.split()
rec = cv2.VideoCapture(0)
model = tf.keras.models.load_model('gesture/gesture_model1.h5')
modelx = tf.keras.models.load_model('point/index_point_x4.h5')
modely = tf.keras.models.load_model('point/index_point_y4.h5')
pts=[]
while True:
    _,frame = rec.read()
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.rectangle(frame,(10,100),(320,380),(0,0,0),2)
    cv2.imshow("VideoInput",frame)
    hand = frame[100:380, 10:320]
    reference = cv2.resize(hand, (300,300))
    img = cv2.resize(hand, (300,300))
    img1 = binaryMask(img)
    img = np.stack((img1,)*3, axis=-1)
    array = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(array, axis=0)
    x=x/255
    pred = list(model.predict(x)[0])
    ind = pred.index(max(pred))
    print(classes[ind])
    array = tf.keras.preprocessing.image.img_to_array(img1)
    x = np.expand_dims(array, axis=0)
    x=x/255
    xcoor = int(modelx.predict(x)[0][0]*300)
    ycoor = int(modely.predict(x)[0][0]*300)
    pts = queuepush(pts,xcoor,ycoor)
    fin = cv2.circle(reference, (ycoor, xcoor), 7, (0,0,0), 3)
    for i in range(len(pts)):
        fin = cv2.circle(reference, pts[i], 1, (0,255,255), 2)
    cv2.imshow("fps", fin)
    
rec.release()
cv2.destroyAllWindows()

