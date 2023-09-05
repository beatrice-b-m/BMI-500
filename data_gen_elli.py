import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm
import pydicom
from skimage.draw import ellipse

# In summary, the code seems well-structuredm and functional for its purpose.
# My review advice would be the consideration of adding docstrings to the functions to explain their purpose, parameters, 
# and return values. This is especially crucial for more complex functions.

def check_side(image):
    slice_l = image[:, :100]
    slice_r = image[:, -100:]
    mean_l = slice_l.mean()
    mean_r = slice_r.mean()
    if mean_l > mean_r:
        return 'L'
    elif mean_r > mean_l:
        return 'R'
    else:
        print('Error: Laterality could not be determined!')
        return 'E' # this may not be easy to understand,  Perhaps raising an exception or using logging would be more appropriate.


def extract_roi(roi: str):  # convert ROI string to list
    roi = roi.translate({ord(c): None for c in "][)(,"})
    roi = list(map(int, roi.split()))
    roi_list = []
    for i in range(len(roi) // 4):
        roi_list.append(roi[4*i:4*i+4])
    return roi_list

# This function name might be  bit general, I would kindly suggest specifying more such as 'save_image_with_mask'
def save_scan(npy, img_dir, filename):
    # add more explanation about input data structure, what does npy mean here?
    # Save array
     # might use os.path.join() for more reliability
    np.save(os.path.join(img_dir,filename[:-4],'.npy'),npy)
   

def get_mask(img, roi_list: list):
    # straightforward and easy to read :) may can add error handling? just a thought 
    mask_shape = (img.shape[0], img.shape[1])
    mask = np.zeros(mask_shape, dtype=np.uint8)
    for roi in roi_list:

        # get height, width, and center coords
        y_center = (roi[0] + roi[2]) // 2
        x_center = (roi[1] + roi[3]) // 2
        half_h = (roi[2] - roi[0]) // 2
        half_w = (roi[3] - roi[1]) // 2

        # get indexes for ellipse
        ellipse_y, ellipse_x = ellipse(y_center, x_center, half_h, half_w, shape=mask_shape)

        # mask ellipse onto mask array
        mask[ellipse_y, ellipse_x] = 1

    return mask


def save_dataset(mode: str, image_rows: int = 1024, image_cols: int = 832, base_path: str = ''):
    if mode == 'train':
        # There are several hardcoded paths such as /home/careinfolab/unet_mammo/images/.... 
        # This isn't portable and won't work on another machine without changes. 
        # better to make these configurable via function parameters or configuration files.
        df_path = '/home/careinfolab/unet_mammo/images/pos_train.csv'
        img_dir = '/home/careinfolab/unet_mammo/images/pos_norm_elli/train/'
    elif mode == 'test':
        df_path = '/home/careinfolab/unet_mammo/images/pos_test.csv'
        img_dir = '/home/careinfolab/unet_mammo/images/pos_norm_elli/test/'
    else:
        df_path = '/home/careinfolab/unet_mammo/images/pos_val.csv'
        img_dir = '/home/careinfolab/unet_mammo/images/pos_norm_elli/val/'

    df = pd.read_csv(df_path)
    # could be made more efficient. 
    #i terrows() isn't the fastest way to iterate over a dataframe. 
    # Consider using apply or vectorized operations when possible
    for i, data in tqdm(df.reset_index().iterrows()):
        # init output array
        img_npy = np.ndarray((image_rows, image_cols, 2), dtype=np.float32)

        # load image from dicom
        dcm = pydicom.dcmread(base_path + data.anon_dicom_path[26:])
        img = dcm.pixel_array

        # extract roi_list and filename
        roi_list = extract_roi(data.ROI_coords)
        filename = data.png_filename

        # correct image laterality
        img_side = check_side(img)
        if img_side == 'R':
            img = np.fliplr(img)
        elif img_side == 'E':
            continue

        # get label mask
        mask = get_mask(img, roi_list)

        # resize image and mask
        img = cv.resize(img, dsize=(image_cols, image_rows), interpolation=cv.INTER_AREA)
        mask = cv.resize(mask, dsize=(image_cols, image_rows), interpolation=cv.INTER_AREA)

        # normalize image between 0 and 1
        img = img.astype(np.float32)
        img_norm = (img-np.min(img))/(np.max(img)-np.min(img))

        # convert mask dtype
        mask = mask.astype(np.float32)

        # Save the image to channel 0 and the mask to channel 1
        img_npy[:, :, 0] = img_norm
        img_npy[:, :, 1] = mask

        save_scan(img_npy, img_dir, filename)
    print(f"'{mode}' set saved...\n")


save_dataset('train', base_path='/media/careinfolab/CI_Lab')
save_dataset('val', base_path='/media/careinfolab/CI_Lab')
save_dataset('test', base_path='/media/careinfolab/CI_Lab')
