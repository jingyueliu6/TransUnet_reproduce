# how nii.gz file

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

label_filename = '../segmentation_dataset/Training/label/label0001.nii.gz'
data_filename = '../segmentation_dataset/Training/img/img0001.nii.gz'
img2 = nib.load(label_filename)
img1 = nib.load(data_filename)
# print(img2)  # <class 'nibabel.nifti1.Nifti1Image'>
# print(img2.header['db_name'])

w, h, q = img2.dataobj.shape
# OrthoSlicer3D(img2.dataobj).show()
print(w, h, q)
## img: 512 512 147
## lebal: 512 512 147
num = 1
plt.figure(1)
for i in range(0, q-27, 12):
    img_arr = img2.dataobj[::2, ::2, i]  # image size reduced by twice
    plt.subplot(5, 2, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1

plt.figure(2)
num = 1
for i in range(0, q-27, 12):
    img_arr = img1.dataobj[::2, ::2, i]
    plt.subplot(5, 2, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1

plt.show()