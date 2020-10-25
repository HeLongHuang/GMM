import cv2 as cv2
import glob
import os
import shutil


class MOG2_mode():
    def __init__(self):
        self.background_model = cv2.createBackgroundSubtractorMOG2()#混合高斯背景建模算法
    def init_path(self):
        path = "MOG2_OUTPUT"
        if os.path.exists(path):
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            os.mkdir(path)

    def judge_img(self):
        img_list = sorted(glob.glob('WavingTrees/*.bmp'))
        print(img_list)
        for i in range(len(img_list)):
            frame = cv2.imread(img_list[i])
            fgmask = self.background_model.apply(frame)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))  # 去噪
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, element)
            cv2.imwrite("MOG2_OUTPUT/{}.jpg".format(str(i).zfill(3)), fgmask)  # 深坑！！！！！！！！！！


if __name__ == '__main__':
    print("创建背景模型")
    model = MOG2_mode()
    model.__init__()
    model.init_path()
    model.judge_img()
    print("检测结束")



