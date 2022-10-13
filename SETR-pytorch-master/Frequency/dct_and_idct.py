import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

# img = cv2.imread('../data/casia1/train_data/Sp_S_CND_A_pla0016_pla0016_0196.jpg',0) #直接读为灰度图像
# img = Image.open('../data/casia1/train_data/Sp_S_CND_A_pla0016_pla0016_0196.jpg').resize((256, 256)).convert("L")

img = Image.open('../data/1.png').resize((224, 224))

'''
plt.subplot(221)
plt.imshow(img,'gray')
plt.title('origial')
plt.xticks([])
plt.yticks([])
'''

# 变换为tensor
img = np.array(img, dtype=np.float32) / 255
img = torch.Tensor(img)
print(img.shape)
a = np.array([0.299, 0.587, 0.114])
# -----------------
rows, cols, colors = img.shape
print(rows)
SecGrayPic = np.zeros([rows, cols],dtype=float)
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]
for i in range(rows):
    for j in range(cols):
      #SecGrayPic[i][j] = np.dot(a,[R[i,j],G[i,j],B[i,j]])
      SecGrayPic[i][j] = 0.299 * R[i,j] + 0.587 * G[i,j] + 0.114 * B[i,j]
print(SecGrayPic)
plt.imshow(SecGrayPic,'gray')
plt.show()
cv2.imwrite("2.jpg", SecGrayPic*255)


print(a)
# --------------------------------
rows, cols= img.shape
mask = np.ones(img.shape, np.uint8)
# print(mask.shape)

mask[int(rows/2)-60:int(rows/2)+60, int(cols/2)-60:int(cols/2)+60] = 0
# print(mask)
# -------------高通滤波器-------------------
f1 = np.fft.fft2(img)
f1shift = np.fft.fftshift(f1)
f11shift = f1shift*mask
f2shift = np.fft.ifftshift(f11shift) #对新的进行逆变换
img_new = np.fft.ifft2(f2shift)
# ---------------------
# 出来的是复数，无法显示
# 取绝对值：将复数变化成实数
# 取对数的目的为了将数据变化到较小的范围（比如0-255）
img_new = np.abs(img_new)
# 调整大小范围便于显示
img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
#plt.subplot(222)
plt.imshow(img_new)
plt.title('Highpass')
plt.xticks([])
plt.yticks([])

# ----------低通滤波-------------
mask1 = np.zeros(img.shape, np.uint8)
mask1[int(rows/2)-20:int(rows/2)+20, int(cols/2)-20:int(cols/2)+20] = 1
f3shift = f1shift*mask1
f4shift = np.fft.ifftshift(f3shift)
img_low = np.fft.ifft2(f4shift)
img_low = np.abs(img_low)
# 调整大小范围便于显示
img_low = (img_low-np.amin(img_low))/(np.amax(img_low)-np.amin(img_low))
'''
plt.subplot(223)
plt.imshow(img_low,'gray')
plt.title('Lowpass')
plt.xticks([])
plt.yticks([])
'''

# ----------带通滤波-------------
mask5 = np.ones(img.shape, np.uint8)
mask5[int(rows/2)-8:int(rows/2)+8,int(cols/2)-8:int(cols/2)+8] = 0
mask4 = np.zeros(img.shape, np.uint8)
mask4[int(rows/2)-80:int(rows/2)+80,int(cols/2)-80:int(cols/2)+80] = 1

mask3 = mask5*mask4
f5shift = f1shift * mask3
f6shift = np.fft.ifftshift(f5shift)
img_mid = np.fft.ifft2(f6shift)
img_mid = np.abs(img_mid)
img_mid = (img_mid-np.amin(img_mid))/(np.amax(img_mid)-np.amin(img_mid))
# 这里得到就是高频信息
'''
plt.subplot(224)
plt.imshow(img_mid,'gray')
plt.title('Midpass')
plt.xticks([])
plt.yticks([])
'''

# plt.show()





'''
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到较小的范围（比如0-255）
ph_f = np.angle(f)
ph_fshift = np.angle(fshift)

plt.subplot(121)
plt.imshow(ph_f, 'gray')
plt.title('original')
plt.subplot(122)
plt.imshow(ph_fshift, 'gray')
plt.title('center')
plt.show()
'''
