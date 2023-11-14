#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 01:38:04 2023

@author: Shiva_roshanravan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


inp_path = '/home/Shiva_roshanravan/Documents/FDECA/Test/Denzel/inpainting.png'
uv_path = '/home/Shiva_roshanravan/Documents/FDECA/Test/Denzel/Denzel.png'

inp_mask = cv2.imread(inp_path)
ret,thresh1 = cv2.threshold(inp_mask,127,255,cv2.THRESH_BINARY)
uv_map = cv2.imread(uv_path)

inp = np.resize(thresh1/255, (256,256,3))
inp = (thresh1/255).astype(np.uint8)
aa = uv_map * (1-inp)


b = aa[153,83]
d = (143,154,193)

cv2.circle(aa, (162,128), 5, (0,0,255), 2)

f = np.where(aa<5, b, aa)
f = np.where(aa<5, d, aa)



inp_mask = cv2.imread(inp_path, cv2.IMREAD_GRAYSCALE)
ret,thresh1 = cv2.threshold(inp_mask,127,255,cv2.THRESH_BINARY)

texture = cv2.inpaint(aa, inp_mask , inpaintRadius=100, flags=cv2.INPAINT_TELEA)
texture = cv2.inpaint(texture, inp_mask , inpaintRadius=10, flags=cv2.INPAINT_TELEA)
texture = cv2.inpaint(texture, inp_mask , inpaintRadius=50, flags=cv2.INPAINT_TELEA)


plt.imshow(aa[:,:,::-1])
plt.imshow(f[:,:,::-1])
plt.imshow(uv_map[:,:,::-1])
plt.imshow(texture[:,:,::-1])
plt.imshow(inp_mask)
#%%
aaa = cv2.Canny(inp_mask, 70, 135)


a = np.where(inp_mask>240)
cv2.circle(inp_mask, (a[0][55],a[1][55]), 5, 0, 10)
plt.imshow(aaa)
