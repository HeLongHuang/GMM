import  matplotlib.pyplot as plt
import matplotlib.image
import glob
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

fig,arrx = plt.subplots(2,2)
image_gmmpri = matplotlib.image.imread('GMM_OUTPUT_Primordial/249.jpg')
image_gmm = matplotlib.image.imread('GMM_OUTPUT/249.jpg')
image_MOG2 = matplotlib.image.imread('MOG2_OUTPUT/249.jpg')
image = matplotlib.image.imread('WavingTrees/b00249.bmp')



arrx[0][0].imshow(image_gmmpri,cmap='gray')
arrx[0][0].set_title('GMM')
arrx[0][1].imshow(image_gmm,cmap='gray')
arrx[0][1].set_title('GMM降噪')
arrx[1][0].imshow(image_MOG2,cmap='gray')
arrx[1][0].set_title('MOG2')
arrx[1][1].imshow(image,cmap='gray')
arrx[1][1].set_title('原图')

fig.show()



R = []
G = []
B = []
img_list = glob.glob('WavingTrees/*.bmp')  # 读取生成的图片

for image_path in img_list:
    image = matplotlib.image.imread(image_path)
    print(image.shape)
    RNUM = image[10, 10, 0]
    R.append(RNUM)
    GNUM = image[10, 10, 1]
    G.append(GNUM)
    BNUM = image[10, 10, 2]
    B.append(BNUM)

plt.plot(R,label='R',color='red')
plt.plot(G,label='G',color='green')
plt.plot(B,label='B',color='blue')
plt.title("图片RGB通道像素采样")
plt.legend()
plt.show()
