import argparse
import numpy as np
import cv2
import math
def gaussian_kernel(size,sigma = 0.5):
	x,y = np.mgrid[(-size//2) + 1:size//2 + 1, (-size//2) + 1:size//2 + 1]
	g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
	return g/g.sum()
def fourier(image,D0,High = False,Gray = False):
	if not Gray:
		(B, G, R) = cv2.split(image)
		x,y = B.shape
	else:
		x,y = image.shape
	if x % 2 == 0:
		center_x = x/2
	else:
		center_x = x/2 + 1
	if y % 2 == 0:
		center_y = y/2
	else:
		center_y = y/2 + 1
	gaussian_array = np.zeros((x,y))
	for i in range(0,x):
		for j in range(0,y):
			gaussian_array[i][j] = math.exp(-1.0 * ((i - center_x)**2 + (j - center_y)**2) / (2 * D0**2))
	if(High):
		gaussian_array = 1 - gaussian_array
	if not Gray:
		BF = np.fft.fft2(B)
		GF = np.fft.fft2(G)
		RF = np.fft.fft2(R)
		Bshift = np.fft.fftshift(BF)
		Gshift = np.fft.fftshift(GF)
		Rshift = np.fft.fftshift(RF)
		BO = Bshift * gaussian_array
		GO = Gshift * gaussian_array
		RO = Rshift * gaussian_array
		BR = np.fft.ifftshift(BO)
		BI = np.fft.ifft2(BR)
		GR = np.fft.ifftshift(GO)
		GI = np.fft.ifft2(GR)
		RR = np.fft.ifftshift(RO)
		RI = np.fft.ifft2(RR)
		out = cv2.merge([np.real(BI),np.real(GI),np.real(RI)])
	else:
		BF = np.fft.fft2(image)
		Bshift = np.fft.fftshift(BF)
		BO = Bshift * gaussian_array
		BR = np.fft.ifftshift(BO)
		out = np.real(np.fft.ifft2(BR))
	return out
#for argparse
argp = argparse.ArgumentParser()
argp.add_argument("-i",dest = "image", nargs = 2, required = True, help = "輸入的圖片路徑。")
argp.add_argument("-s",dest = "size", nargs = 2,help = "kernel大小，除了傅立葉轉換都需要有，因low pass都是使用Gaussian。")
argp.add_argument("-o",dest = "output", nargs = 1, required = True,help = "輸出檔案名稱。")
argp.add_argument("-sl",dest = "sigma", nargs = 2, help = "sigma值。(low high)，若是傅立葉轉換的話為low和high的頻率")
argp.add_argument("-f",dest = "fourier",default = False,action = 'store_true',help = "使用傅立葉變換,最優先。")
argp.add_argument("-d",dest = "downscale",default = False,action = 'store_true',help = "降畫素，預設是三倍。")
argp.add_argument("-la",dest = "laplacian",default = False,action = 'store_true',help = "high pass使用laplacian filter，優先度第二。")
argp.add_argument("-sb",dest = "sobel",default = False,action = 'store_true',help = "high pass使用sobel filter(x)，優先度第三。")
parser = argp.parse_args()
Input = parser.image
Size = parser.size
output = parser.output
Sigma = parser.sigma
F = parser.fourier
D = parser.downscale
L = parser.laplacian
SB = parser.sobel
if not F:
	if Size is None:
		print('Size cannot be empty!')
		exit()
	elif int(Size[0]) % 2 != 1 or int(Size[1]) % 2 != 1:
 		print("Size must be odd!")
 		exit()
#openfiles
img1 = cv2.imread(Input[0]) 
img2 = cv2.imread(Input[1]) 
#縮小圖片用
if D:
	w,h,r = img1.shape
	print(w,h,r)
	img1 = cv2.resize(img1,(h//3,w//3),interpolation=cv2.INTER_AREA)
	img2 = cv2.resize(img2,(h//3,w//3),interpolation=cv2.INTER_AREA)
#mainfunction
if F:
	print("Use fourier.")
	if Sigma is None:
		print("No frequency.")
		exit() 
	Low = fourier(img1,int(Sigma[0]))
	Low = np.real(Low)
	Low = np.rint(Low)
	High = fourier(img2,int(Sigma[1]),High = True)
	High = np.real(High)
	High = np.rint(High)
	outp = (High + Low)/255
	[n,h,k] = outp.shape
	for i in range(n):
		for j in range(h):
			for r in range(k):
				if outp[i][j][r] < 0:
					outp[i][j][r] = 0
				if High[i][j][r] < 0:
					High[i][j][r] = 0
				if Low[i][j][r] < 0:
					Low[i][j][r] = 0

	cv2.imshow('My Image', outp)
	cv2.imshow('Low Image', Low/255)
	cv2.imshow('High Image', High/255)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite(output[0],np.rint(outp*255)) 
elif L:
	imgl = cv2.filter2D(img1,-1,gaussian_kernel(int(Size[0]),100))
	imgh = cv2.Laplacian(img2,cv2.CV_64F)
	cv2.imshow('My Image', imgh + imgl)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite(output[0],imgh + imgl)
elif SB:
	imgl = cv2.filter2D(img1,-1,gaussian_kernel(int(Size[0]),10))
	imgh = cv2.Sobel(img2,cv2.CV_64F,1,1,ksize=5)
	cv2.imshow('My Image', imgh + imgl)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite(output[0],imgh + imgl)
else:
	if parser.sigma is not None:
		imgl = cv2.filter2D(img1,-1,gaussian_kernel(int(Size[0]),float(Sigma[0])))
		imgh = img2 - cv2.filter2D(img2,-1, gaussian_kernel(int(Size[1]),float(Sigma[1])))
	else:
		imgl = cv2.filter2D(img1,-1,gaussian_kernel(int(Size[0]),1))
		imgh = img2 - cv2.filter2D(img2,-1, gaussian_kernel(int(Size[1]),1))
	out = imgl + imgh
	cv2.imshow('My Image', imgl + imgh)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite(output[0],imgh + imgl)
	#cv2.imwrite('imgl.jpg',imgl)
	#cv2.imwrite('imgh.jpg',imgh)

