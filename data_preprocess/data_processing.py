import nibabel as nib
import os
import numpy as np

img_path = '../segmentation_dataset/Training/img/'
seg_path = '../segmentation_dataset/Training/label/'
saveimg_path = '../segmentation_dataset/npy_train/'
saveseg_path = '../segmentation_dataset/npy_seg/'

img_names = os.listdir(img_path)
seg_names = os.listdir(seg_path)

# img_name = img_names[0]
# img = nib.load(img_path + img_name).get_data()  # loading
# img = np.array(img)
# print(img.shape)
# print(np.max(img))  # 2976
# print(np.min(img))  # -1024
# seg_name = seg_names[0]
# seg = nib.load(seg_path + seg_name).get_data()  # loading
# seg = np.array(seg)
# print(np.max(seg))  # 13
# '''
# (1) spleen
# (2) right kidney
# (3) left kidney
# (4) gallbladder
# (5) esophagus
# (6) liver
# (7) stomach
# (8) aorta
# (9) inferior vena cava
# (10) portal vein and splenic vein
# (11) pancreas
# (12) right adrenal gland
# (13) left adrenal gland
# '''
# print(np.min(seg))  # 0
# for img_name in img_names:
#     print(img_name)
#     img = nib.load(img_path + img_name).get_data()  # loading
#     img = np.array(img)
#     print(img.shape)
#     print("--------------------------------------------")
# for seg_name in seg_names:
#     print(seg_name)
#     seg = nib.load(seg_path + seg_name).get_data()  # loading
#     seg = np.array(seg)
#     print(seg.shape)
#     print("--------------------------------------------")
#  the size of images is different, so they should be clipped.
slice_selection = {}  # slice which has label
for seg_name in seg_names:
    print(seg_name)
    seg = nib.load(seg_path + seg_name).get_data()  # loading
    seg = np.array(seg)
    slice = []
    for i in range(20, seg.shape[2], 5):
        l1 = np.sum(seg[:, :, i] == 2)  # left kidney
        l2 = np.sum(seg[:, :, i] == 3)  # right kidney (Some patients may not have (2) right kidney or (4) gallbladder)
        l3 = np.sum(seg[:, :, i] == 6)  # liver
        if l1 > 5 and l3 > 10:
            slice.append(i)
            seg_resize = seg[::2, ::2, i]
            # print(seg_resize.shape)
            seg_resize = seg_resize[:224, :224]
            # print(seg_resize.shape)
            np.save(saveseg_path + str(seg_name).split('.')[0]+str(i)+'npy', seg_resize)  # saving
    slice_selection['img' + str(seg_name[5:9])] = slice
# print(slice_selection)
# {'img0004': [80, 85, 90, 95], 'img0035': [55, 60, 65, 70], 'img0023': [60, 65, 70, 75], 'img0027': [45, 50, 55, 60], 'img0031': [25, 30, 35, 40, 45], 'img0003': [125, 130, 135, 140], 'img0029': [50, 55], 'img0002': [80, 85, 90, 95, 100], 'img0021': [85, 90, 95, 100, 105, 110], 'img0036': [105, 110, 115, 120, 125, 130, 135, 140], 'img0005': [55, 60], 'img0028': [45, 50, 55, 60], 'img0026': [65], 'img0030': [85, 90, 95, 100, 105, 110, 115], 'img0022': [50, 55, 60, 65, 70], 'img0008': [95, 100, 105, 110], 'img0040': [80, 85, 90, 95, 100, 105, 110], 'img0006': [75, 80, 85, 90, 95], 'img0001': [80, 85, 90, 95, 100, 105, 110, 115], 'img0039': [50, 55, 60, 65], 'img0007': [85, 90, 95, 100, 105, 110], 'img0010': [75, 80, 85, 90, 95, 100, 105], 'img0024': [75, 80, 85, 90, 95, 100], 'img0033': [60, 65, 70], 'img0037': [55, 60, 65, 70, 75], 'img0032': [80, 85, 90, 95, 100, 105], 'img0009': [55, 60, 65, 70, 75, 80, 85, 90], 'img0038': [55, 60, 65, 70], 'img0025': [45, 50], 'img0034': [65]}

for img_name in img_names:
    print(img_name)
    img = nib.load(img_path + img_name).get_data()  # loading
    img = np.array(img)
    slice = slice_selection[str(img_name).split('.')[0]]
    for i in slice:
        img_select = img[::2, ::2, i]
        img_resize = img_select[:224, :224]
        img_resize = np.clip(img_resize, -125, 275)
        img_resize = (img_resize + 125)/400
        # print(str(img_name).split('.')[0]+str(i))
        np.save(saveimg_path + str(img_name).split('.')[0]+str(i)+'npy', img_resize)  # saving


# for img_name in img_names:
#     print(img_name)
#     img = nib.load(img_path + img_name).get_data()  # loading
#     img = np.array(img)
#     img = np.clip(img, -125, 275)
#     img = (img + 125)/400
#     for i in range(0, img.shape[2], 20):
#         np.save(saveimg_path + str(img_name).split('.')[0]+str(i)+'npy', img[:, :, i])  # saving
#
# for seg_name in seg_names:
#     print(seg_name)
#     seg = nib.load(seg_path + seg_name).get_data()  # loading
#     seg = np.array(seg)
#     for i in range(0, seg.shape[2], 20):
#         np.save(saveseg_path + str(seg_name).split('.')[0]+str(i)+'npy', seg[:, :, i])  # saving

