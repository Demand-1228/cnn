from imutils import contours
import numpy as np
import argparse
import cv2
import myutils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='images\credit_card_04.png', help="path to input image")
ap.add_argument("-t", "--template", default='ocr_a_reference.png' ,help="path to template OCR-A image")
args = vars(ap.parse_args())
def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

img = cv2.imread(args["image"])
img = myutils.resize(img, width=300)
#cv_show('resize',img)
kernel = np.ones((3,9),np.uint8)
print (kernel)
kernel1 = np.ones((5,5),np.uint8) 
print (kernel1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
cv_show('tophat',tophat)
#ref = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]



sobelx = cv2.Sobel(tophat,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
(minVal, maxVal) = (np.min(sobelx), np.max(sobelx))
sobelx = (255 * ((sobelx - minVal) / (maxVal - minVal)))
sobelx = sobelx.astype("uint8")
cv_show('sobelx',sobelx)
closing = cv2.morphologyEx(sobelx, cv2.MORPH_CLOSE, kernel) 
ref_grad = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv_show('2',ref_grad)

#opening = cv2.morphologyEx(ref, cv2.MORPH_OPEN, kernel) #开
#tophat = cv2.morphologyEx(ref, cv2.MORPH_TOPHAT, kernel) #礼帽
closing = cv2.morphologyEx(ref_grad, cv2.MORPH_CLOSE, kernel) #闭操作
#dilate = cv2.dilate(ref,kernel,iterations = 5)

#sobely = cv2.Sobel(tophat,cv2.CV_64F,0,1,ksize=3)
#sobely = cv2.convertScaleAbs(sobely) 
#sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0) 
# cv_show('sobelx',sobelx)
closing1 = cv2.morphologyEx(ref_grad, cv2.MORPH_CLOSE, kernel1) #闭操作
#closing1 = cv2.morphologyEx(closing1, cv2.MORPH_CLOSE, kernel)

#res = np.hstack((ref,tophat,sobelxy,closing1))
cv_show('ref',closing1)


#refCnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
#res = cv2.drawContours(img.copy(),refCnts,-1,(0,255,0),5) #画白色的轮廓