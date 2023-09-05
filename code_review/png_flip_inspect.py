import pandas as pd
import cv2 as cv
import numpy as np
import pydicom
import threading
import os
from time import sleep

"""
this is a script used to check a png and dicom and determine 
whether the png has been flipped at some stage in preprocessing.
if it has, the region of interest (roi) is corrected for the dicom
and recorded to the dataframe.
"""


class LateralityChecker:
    def __init__(self, df):
        self.df = df
        self.out_df = None
        self.logger_active = True
        self.save_dataframe = False
        
    def start(self, chunk_size: int = 10_000, sleep_time: int = 30, 
              log_path: str = './meta_laterality_check_df.pickle'):
        print(f'logging dataframe to {log_path}')
        
        # start the generator thread
        generator = threading.Thread(
            target=self.check_df_img_sides, 
            args=[chunk_size]
        )
        print('starting generator thread...')
        generator.start()
        
        # start the logger thread
        logger = threading.Thread(
            target=self.dataframe_logger, 
            args=[sleep_time, log_path]
        )
        print('starting logger thread...')
        logger.start()            
        
        # join threads
        generator.join()
        logger.join()
        
    def dataframe_logger(self, sleep_time: int, log_path: str):
        while self.logger_active:
            print('logger active...')
            sleep(sleep_time)
            
            # if save dataframe signal received
            if self.save_dataframe:
                print(f'logging out_df to {log_path}')
                self.out_df.to_pickle(log_path)
                
                # reset signal
                self.save_dataframe = False
                
        print(f'logging final out_df to {log_path}')
        self.out_df.to_pickle(log_path)

    def check_df_img_sides(self, chunk_size: int):
        self.out_df = self.df.copy()

        # make png_flipped column
        self.out_df.loc[:, 'png_flipped'] = np.nan
        self.out_df.loc[:, 'dicom_roi_coords'] = ''

        # iterate over dataframe
        for i, data in tqdm(self.df.iterrows(), total=len(self.df)):
            # if chunk is completed, save df
            if i % chunk_size == 0:
                # send save dataframe signal
                self.save_dataframe = True

            # get png and dcm paths
            png_path = data.png_path
            dcm_path = data.anon_dicom_path

            # load png and dicom
            png = cv.imread(png_path)
            dcm = pydicom.dcmread(dcm_path)
            dcm_img = dcm.pixel_array

            # load roi
            if len(data.num_roi) > 0:
                roi_list = extract_roi(data.ROI_coords)
            else:
                roi_list = None

            # check png and dcm side
            png_side = check_side(png[:, :, 0])
            dcm_side = check_side(dcm_img)
            
            # if lateralities can't be determined, set to nan
            if (dcm_side == 'E') | (png_side == 'E'):
                self.out_df.loc[i, 'png_flipped'] = np.nan
                self.out_df.loc[i, 'dicom_roi_coords'] = 'ERROR'
            
            # else if dcm and png sides don't match, flipped is true
            elif dcm_side != png_side:
                self.out_df.loc[i, 'png_flipped'] = True
                if roi_list is not None:
                    self.out_df.loc[i, 'dicom_roi_coords'] = str(switch_coord_side(roi_list, png.shape))
                
            # else flipped is false
            else:
                self.out_df.loc[i, 'png_flipped'] = False
                if roi_list is not None:
                    self.out_df.loc[i, 'dicom_roi_coords'] = str(roi_list)

                
def check_side(image, slice_width: int = 20, percentile: int = 2):
    """
    function to check laterality of png/dcm
    """    
    # take slices of the image on the left and right sides of the img
    slice_L = image[:, :slice_width]
    slice_R = image[:, -slice_width:]
    
    # get the background threshold by finding a percentile of the img
    bg_threshold = np.percentile(image, percentile)
    
    # count number of background pixels on each side
    num_bg_L = (np.array(slice_L) <= bg_threshold).sum()
    num_bg_R = (np.array(slice_R) <= bg_threshold).sum()
    
    # if there are fewer background pixels on the left than the right
    # laterality is left
    if num_bg_L < num_bg_R:
        return 'L'
    
    # if there are fewer background pixels on the right than the left
    # laterality is right
    elif num_bg_R < num_bg_L:
        return 'R'
    
    # if num background pixels is equal, laterality can't be determined
    else:
        print('Error: Laterality could not be determined!')
        return 'E'


def switch_coord_side(roi_list, full_shape):
    """
    function to switch the side of an roi
    """
    roi_list_out = []
    
    for roi in roi_list:
        minX = roi[1]
        maxX = roi[3]
        roi[1] = full_shape[1] - maxX
        roi[3] = full_shape[1] - minX
        roi_list_out.append(roi)
        
    return roi_list_out


def extract_roi(roi: str):  
    """
    function to strip punction from an ROI string and 
    convert it to a list
    """
    roi = roi.translate({ord(c): None for c in "][)(,"})
    roi = list(map(int, roi.split()))
    roi_list = []
    for i in range(len(roi) // 4):
        roi_list.append(roi[4*i:4*i+4])
    return roi_list


print('loading metadata...')
meta_path = '/PATH/REDACTED/metadata.pickle'
meta_df = pd.read_pickle(meta_path)

print('updating png paths...')
# update png paths for PACS
meta_df.loc[:, 'png_path'] = meta_df['png_path'].str.replace('/PATH/REDACTED/',
                                                             '/PATH/REDACTED/')

# load laterality checker object and run it
print('running laterality checker...')
lat_checker = LateralityChecker(meta_df)
lat_checker.start()

print('script complete.')