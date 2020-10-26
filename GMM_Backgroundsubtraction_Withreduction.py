import numpy as np
import cv2 as cv2
import glob
import time
import os
import shutil

global m_fit_num

class GMM():

    def __init__(self):
        self.GMM_MAX_COMPONT = 5 # 混合高斯数
        self.SIGMA = 30
        self.WEIGHT = 0.05
        self.T = 0.7  # 模型排序判读阀值
        self.alpha = 0.005   # 学习率
        self.eps = pow(10, -10)
        self.channel = 3  # RGB三个通道
        self.m_weight = [[] for i in range(self.GMM_MAX_COMPONT * self.channel)]  # 权重
        self.m_mean = [[] for i in range(self.GMM_MAX_COMPONT * self.channel)]  # 均值
        self.m_sigma = [[] for i in range(self.GMM_MAX_COMPONT * self.channel)]  # 方差

    def init_model(self,img):
        row , col , channel = img.shape # 得到图片的长宽高 以及其中的通道数
        global m_fit_num
        for i in range(self.GMM_MAX_COMPONT * self.channel):
            self.m_weight[i] = np.zeros((row,col),dtype="float32")  # 每个点有5个高斯模型，总共三个通道
            self.m_mean[i] = np.zeros((row, col), dtype='float32')
            self.m_sigma[i] = np.ones((row, col), dtype='float32')
            self.m_sigma[i] *= self.SIGMA
        m_fit_num = np.zeros((row,col),dtype="int32")

    def train_model(self,images):
        row, col, channel = images.shape  # 得到图片的长宽高 以及其中的通道数
        B,R,G = cv2.split(images)  # 利用cv2提取图像RGB三个通道的图形矩阵
        m_mask = np.zeros((row,col),dtype=np.uint8)
        m_mask[:] = 255
        for i in range(row):  # 遍历每一个像素点
            for j in range(col):
                cnt = 0
                for c,img in enumerate((B,G,R)):
                    num_fit = 0
                    for k in range(c * self.GMM_MAX_COMPONT,c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT):
                        if self.m_weight[k][i][j] != 0:  # 权重不等于0
                            delta = abs(img[i][j] - self.m_mean[k][i][j])
                            if float(delta) < 2.5 * self.m_sigma[k][i][j]:   # 在2.5个方差之内 平均数 方差 等参数
                                self.m_weight[k][i][j] = (1 - self.alpha) * self.m_weight[k][i][j] + self.alpha * 1
                                self.m_mean[k][i][j] = (1 - self.alpha) * self.m_mean[k][i][j] + self.alpha * img[i][j]
                                self.m_sigma[k][i][j] = np.sqrt((1 - self.alpha) * self.m_sigma[k][i][j] * self.m_sigma[k][i][j] + self.alpha * (img[i][j] - self.m_mean[k][i][j]) * (img[i][j] - self.m_mean[k][i][j]))
                                num_fit += 1
                            else:
                                self.m_weight[k][i][j] *= (1 - self.alpha)

                    for p in range(c * self.GMM_MAX_COMPONT, c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT):   # 对权重进行降序 根据𝜔/𝜎降序排序 等会进行选择
                        for q in range(p + 1, c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT):
                            if (self.m_weight[p][i][j] / self.m_sigma[p][i][j]) <= (self.m_weight[q][i][j] / self.m_sigma[q][i][j]):
                                self.m_sigma[p][i][j], self.m_sigma[q][i][j] = self.m_sigma[q][i][j], self.m_sigma[p][i][j]
                                self.m_weight[p][i][j], self.m_weight[q][i][j] = self.m_weight[q][i][j], self.m_weight[p][i][j]
                                self.m_mean[p][i][j], self.m_mean[q][i][j] = self.m_mean[q][i][j], self.m_mean[p][i][j]
                    if num_fit == 0:  # 没有匹配到任何一个高斯模型
                        if self.m_weight[c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT-1][i][j] ==0 :
                            for kk in range(c * self.GMM_MAX_COMPONT, c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT):
                                if (0 == self.m_weight[kk][i][j]):  # 重新初始化参数
                                    self.m_weight[kk][i][j] = self.WEIGHT
                                    self.m_mean[kk][i][j] = img[i][j]
                                    self.m_sigma[kk][i][j] = self.SIGMA
                                    break
                        else:
                            self.m_weight[c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT - 1][i][j] = self.WEIGHT
                            self.m_mean[c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT - 1][i][j] = img[i][j]
                            self.m_sigma[c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT - 1][i][j] = self.SIGMA

                    weight_sum = 0  # 每个高斯模型的权重要进行归一化操作
                    for nn in range(c * self.GMM_MAX_COMPONT, c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT):
                        if self.m_weight[nn][i][j] != 0:
                            weight_sum += self.m_weight[nn][i][j]
                        else:
                            break
                    weight_scale = 1.0 / (weight_sum + self.eps)
                    weight_sum = 0

                    for nn in range(c * self.GMM_MAX_COMPONT, c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT):
                        if self.m_weight[nn][i][j] != 0:
                            self.m_weight[nn][i][j] *= weight_scale
                            weight_sum += self.m_weight[nn][i][j]
                            if abs(img[i][j] - self.m_mean[nn][i][j]) < 2 * self.m_sigma[nn][i][j]:
                                cnt += 1
                                break
                            if weight_sum > self.T:
                                if abs(img[i][j] - self.m_mean[nn][i][j]) < 2 * self.m_sigma[nn][i][j]:
                                    cnt += 1
                                break
                        else:
                            break
                if cnt == channel:
                    m_mask[i][j] = 0

        m_mask = cv2.medianBlur(m_mask, 7)
        kernel_d = np.ones((5, 5), np.uint8)
        m_mask = cv2.dilate(m_mask, kernel_d)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 调用库函数开启形态学去噪  椭圆形状
        m_mask = cv2.morphologyEx(m_mask, cv2.MORPH_CLOSE, element)  # 开运算去噪

        return m_mask

    def judge_img(self,imgs):
        row, col, channel = imgs.shape
        B, G, R = cv2.split(imgs)
        m_mask = np.zeros((row, col), dtype=np.uint8)
        m_mask[:] = 255
        for i in range(row):
            for j in range(col):
                cnt = 0
                for c, img in enumerate((B, G, R)):       # 一张图片的每个像素点进行判断  是否是作为前景还是背景
                    weight_sum = 0
                    for nn in range(c * self.GMM_MAX_COMPONT, c * self.GMM_MAX_COMPONT + self.GMM_MAX_COMPONT):
                        if self.m_weight[nn][i][j] != 0:
                            weight_sum += self.m_weight[nn][i][j]
                            if abs(img[i][j] - self.m_mean[nn][i][j]) < 2 * self.m_sigma[nn][i][j]:
                                cnt += 1
                                break
                            if weight_sum > self.T:
                                if abs(img[i][j] - self.m_mean[nn][i][j]) < 2 * self.m_sigma[nn][i][j]:
                                    cnt += 1
                                break
                        else:
                            break

                if cnt == channel:
                    m_mask[i][j] = 0

        m_mask = cv2.medianBlur(m_mask, 7)
        kernel_d = np.ones((5, 5), np.uint8)
        m_mask = cv2.dilate(m_mask, kernel_d)
        return m_mask




if __name__ == '__main__':
    file_list = glob.glob('WavingTrees/b*.bmp')    # 读入测试文件得列表
    GMM_Model = GMM()
    GMM_Model.__init__()  # 初始化模型
    path = "GMM_OUTPUT"
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)
    i = -1
    for file in file_list:
        i += 1
        img = cv2.imread(file)
        if i == 0:
            GMM_Model.init_model(img)  # 第一张图片
        if i <= 200:                # 前面的200张用于训练模型
            t1 = time.time()
            print("第{}次训练".format(i))
            m_mask = GMM_Model.train_model(img)
            t2 = time.time()
            print("花费时间：",t2 - t1)
        if i == 286:    # 训练完毕 开始识别
            print("开始背景检测")
            t1 = time.time()
            j = 0
            for temp_file in file_list:
                temp_img = cv2.imread(temp_file)
                m_mask = GMM_Model.judge_img(temp_img)
                cv2.imwrite("GMM_OUTPUT/{}.jpg".format(str(j).zfill(3)), m_mask)
                j += 1
            t2 = time.time()
            print("检测花费时间：",t2 - t1)







