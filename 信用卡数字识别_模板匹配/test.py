from imutils import contours
import numpy as np
import argparse
import cv2
import myutils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='images\credit_card_01.png', help="path to input image")
ap.add_argument("-t", "--template", default='ocr_a_reference.png' ,help="path to template OCR-A image")
args = vars(ap.parse_args())

# 绘图展示
def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
# 读取一个模板图像
img = cv2.imread(args["template"])
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
# 计算轮廓
#cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
#返回的list中每个元素都是图像中的一个轮廓

refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
#print (refCnts[0][0][0])
cv2.drawContours(img,refCnts,-1,(0,255,0),5) #画白色的轮廓
cv_show('img',img)

boundingBoxes = sorted([cv2.boundingRect(c) for c in refCnts], 
                        key=lambda b:b[0])  

digits = {}
# 遍历每一个轮廓
for i,b in enumerate(boundingBoxes):
    (x, y, w, h) = b
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi
    cv_show(str(i),roi)
    
    
