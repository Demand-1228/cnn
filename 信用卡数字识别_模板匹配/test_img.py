from imutils import contours
import numpy as np
import argparse
import cv2
import myutils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='images\credit_card_05.png', help="path to input image")
ap.add_argument("-t", "--template", default='ocr_a_reference.png' ,help="path to template OCR-A image")
args = vars(ap.parse_args())
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
    #cv_show(str(i),roi)

img = cv2.imread(args["image"])
img = cv2.resize(img,(800,500))
#cv_show('resize',img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv_show('gray',gray)
ref = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

kernel = np.ones((5,5),np.uint8) 
#opening = cv2.morphologyEx(ref, cv2.MORPH_OPEN, kernel) #开
#tophat = cv2.morphologyEx(ref, cv2.MORPH_TOPHAT, kernel) #礼帽
dilate = cv2.dilate(ref,kernel,iterations = 5)
res = np.hstack((ref,dilate))
#cv_show('ref',res)


refCnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
res = cv2.drawContours(img.copy(),refCnts,-1,(0,255,0),5) #画白色的轮廓
#cv_show('img',res)

boundingBoxes = [cv2.boundingRect(c) for c in refCnts]
boundingBoxes = sorted(boundingBoxes, key = lambda b :b[0])
target={}
box= {}
j = 0
#print(boundingBoxes)
for i,b in enumerate(boundingBoxes):
    ar = b[2]/float(b[3])
    
    if ar>2.5 and ar<3.0 and b[2]<200:
        
        (x, y, w, h) = b
        box[j] = b
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        target[j] = ref[y:y + h, x:x + w]
        target[j] = cv2.threshold(target[j], 10, 255, cv2.THRESH_BINARY)[1]
        #cv_show('img',target[j])
        print(target[j].shape)
        cv2.imwrite('%d.png'%j,target[j])
        j = j + 1
#for i,t in enumerate(target):
output=[]
for t in range(len(target)):
    refCnts = cv2.findContours(target[t], cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

    for c in refCnts:
        scores = []
        (x, y, w, h) = cv2.boundingRect(c)
        ref = target[t][y:y + h, x:x + w]
        ref = cv2.resize(ref, (57, 88))
        cv_show('img',ref)
        for i in range(10):
            result = cv2.matchTemplate(ref, digits[i], cv2.TM_CCOEFF)
            (_, score,_, _) = cv2.minMaxLoc(result)
            scores.append(score)
        output.append(scores.index(max(scores)))

print(output)   



