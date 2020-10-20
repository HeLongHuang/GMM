import cv2
import glob
import matplotlib.image
import numpy as np
import os
fgbg = cv2.createBackgroundSubtractorMOG2()#混合高斯背景建模算法

img_list = sorted(glob.glob('WavingTrees/*.bmp'))

for i in range(len(img_list)):
    frame = cv2.imread(img_list[i])
    fgmask = fgbg.apply(frame)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))  # 形态学去噪
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)  # 开运算去噪
    cv2.imwrite("test1/{}.jpg".format(str(i).zfill(3)), fgmask)


img_list1 = glob.glob('test1/*.jpg')
print(img_list1)
img_list1 = sorted(img_list1)
print(img_list1)
# filePath = 'test1'
# name = sorted(os.listdir(filePath))
# print(name)

img_list_wavingtress1 = sorted(glob.glob('WavingTrees/b*.bmp'))
print(img_list_wavingtress1)
# print("生成图片的数量是：",len(img_list1))
print("原始的数量是：",len(img_list_wavingtress1))
import cv2
#读取一张图片
img = cv2.imread('test1/000.jpg')
#获取当前图片的信息
imgInfo = img.shape
size = (imgInfo[1],imgInfo[0])
print("图片的大小为：",imgInfo)
print("定义的size：",size)

# 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
# videowrite = cv2.VideoWriter('test1.mp4',-1,10,size)
# for i in img_list1:
#     img = cv2.imread(i)
#     videowrite.write(img)
# videowrite = cv2.VideoWriter('test2.mp4',-1,10,size)
# for i in img_list_wavingtress1:
#     img = cv2.imread(i)
#     videowrite.write(img)

videowrite = cv2.VideoWriter("test.mp4",-1,10,(320,120))
for i in range(len(img_list)):
    imgleft = cv2.imread(img_list1[i])
    imgright = cv2.imread(img_list_wavingtress1[i])
    newimage = np.concatenate((imgright,imgleft), axis=1)
    print(newimage.shape)
    videowrite.write(newimage)


