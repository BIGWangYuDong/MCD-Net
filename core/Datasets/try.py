# data_infos = []
# files_Tgt = []
# x = 'a'
# Tgt_img_paths = []
# with open('/home/dong/python-project/ZZ/XianZhuXing/image/xx', 'r') as f:
#     data_list = f.read().split('\n')
#     for data in data_list:
#         if len(data) == 0:
#             continue
#         data_infos.append({
#             "image": x + data,
#             'gt':x + x + data
#         })
# img_info = data_infos[1]['image']
# gt_info = data_infos[1]['gt']
# results = dict(image_info=img_info, gt_info=gt_info)
# import copy
# results0 = copy.deepcopy(data_infos[1])
# print()

##############################################################################
a = '/home/dong/python-project/ZZ/XianZhuXing/image/underwater/2_img_.png'
b = '/home/dong/python-project/ZZ/XianZhuXing/image/DUTS/DUTS-TR/DUTS-TR-Mask/ILSVRC2012_test_00000004.png'
from PIL import Image
from skimage import io, transform, color
# import torchvision.transforms as transforms
from torchvision import datasets, transforms
import numpy as np

image1 = Image.open(a)
# image3 = image = io.imread(b)
image2 = Image.open(b)
divisor = 32

# print()
# t1 = 256
# t2 = (256, 300)
# if isinstance(t1, int):
#     h, w = t1, t1
# elif isinstance(t2, tuple):
#     h, w = t2
# osize = [h, w]
import numbers
w = image1.size[0]
h = image1.size[1]
# w, h
pad_w = int(np.ceil(image1.size[0] / divisor)) * divisor
pad_h = int(np.ceil(image1.size[1] / divisor)) * divisor
scale_transform = transforms.Resize((pad_h,pad_w))
image11 = scale_transform(image1)

pad_w = int(np.ceil(image1.size[0] / divisor)) * divisor
pad_h = int(np.ceil(image1.size[1] / divisor)) * divisor
padding = (0, 0, pad_w - w, pad_h - h)
if isinstance(padding, tuple) and len(padding) in [2, 4]:
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
elif isinstance(padding, numbers.Number):
    padding = (padding, padding, padding, padding)
else:
    raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                     f'But received {padding}')
scale_transform_src = transforms.Pad(padding)
h, w = image2.size

src_img1 = scale_transform_src(image1)
src_img2 = scale_transform_src(image2)
print()