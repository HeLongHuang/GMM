import glob
import cv2
import numpy as np


def generate_video_mog2():
    img_list_mog2 = glob.glob('MOG2_OUTPUT/*.jpg')  # 读取生成的图片
    print(img_list_mog2)
    img_list_mog2 = sorted(img_list_mog2)

    img_list_wavingtress = sorted(glob.glob('WavingTrees/b*.bmp'))
    print(img_list_wavingtress)
    # 读取一张图片
    img = cv2.imread('MOG2_OUTPUT/000.jpg')
    # 获取当前图片的信息
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    print("图片的大小为：", imgInfo)
    print("定义的size：", size)

    videowrite = cv2.VideoWriter("With_MOG2.mp4", -1, 10, (320, 120))
    for i in range(len(img_list_mog2)):
        imgleft = cv2.imread(img_list_mog2[i])
        imgright = cv2.imread(img_list_wavingtress[i])
        newimage = np.concatenate((imgright, imgleft), axis=1)
        videowrite.write(newimage)



def generate_video_gmm():
    img_list_gmm = glob.glob('GMM_OUTPUT/*.jpg')  # 读取生成的图片
    print(img_list_gmm)
    img_list_mog2 = sorted(img_list_gmm)

    img_list_wavingtress = sorted(glob.glob('WavingTrees/b*.bmp'))
    print(img_list_wavingtress)
    # 读取一张图片
    img = cv2.imread('GMM_OUTPUT/000.jpg')
    # 获取当前图片的信息
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    print("图片的大小为：", imgInfo)
    print("定义的size：", size)

    videowrite = cv2.VideoWriter("With_GMM.mp4", -1, 10, (320, 120))
    for i in range(len(img_list_mog2)):
        imgleft = cv2.imread(img_list_mog2[i])
        imgright = cv2.imread(img_list_wavingtress[i])
        newimage = np.concatenate((imgright, imgleft), axis=1)
        videowrite.write(newimage)



def generate_video_gmm_primordial():
    img_list_gmm = glob.glob('GMM_OUTPUT_Primordial/*.jpg')  # 读取生成的图片
    print(img_list_gmm)
    img_list_mog2 = sorted(img_list_gmm)

    img_list_wavingtress = sorted(glob.glob('WavingTrees/b*.bmp'))
    print(img_list_wavingtress)
    # 读取一张图片
    img = cv2.imread('GMM_OUTPUT_Primordial/000.jpg')
    # 获取当前图片的信息
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    print("图片的大小为：", imgInfo)
    print("定义的size：", size)

    videowrite = cv2.VideoWriter("With_GMM_Primordial.mp4", -1, 10, (320, 120))
    for i in range(len(img_list_mog2)):
        imgleft = cv2.imread(img_list_mog2[i])
        imgright = cv2.imread(img_list_wavingtress[i])
        newimage = np.concatenate((imgright, imgleft), axis=1)
        videowrite.write(newimage)

if __name__ == '__main__':

    generate_video_gmm()
    generate_video_mog2()
    generate_video_gmm_primordial()


