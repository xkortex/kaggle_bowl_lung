#----------------------------------------
#--- Downsampling routine 
#--- Dan Elton 
#----------------------------------------

import numpy as np 
import dicom
import SimpleITK as sitk
import os, sys
import scipy.ndimage
import glob

import time 
from tqdm import tqdm

import pandas.tools.plotting

#-----------------------------------------------------
def load_scan(path):
    """ Load the scans in a given folder path
        returns: an array with all slices"""
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

#-----------------------------------------------------
def get_pixels_hu(scans):
    """Convert to Houndsworth Units"""
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

#-----------------------------------------------------
def resample_dcm(image, scan, new_spacing=(1, 1, 1), verbose=False):
    '''Resampling to make x-y-z pixel spacing standardized'''
    
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    if verbose: print('Initial Spacing:', spacing)


    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    if verbose: print('Final Spacing:', new_spacing)

    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing


def resample_mhd(rawScan, new_spacing=(1, 1, 1), verbose=False):
    '''Resampling to make x-y-z pixel spacing standardized'''
    image = np.array(sitk.GetArrayFromImage(rawScan), dtype=np.int16)
    spacing = np.array(rawScan.GetSpacing())  # spacing of voxels in world coor. (mm)
    sx, sy, sz = spacing
    spacing = np.array([sz, sx, sy])
    # Determine current pixel spacing
    if verbose: print('Initial Spacing: {} Shape: {}'.format(spacing, image.shape))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor


    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    if verbose: print('Final Spacing:   {} Shape: {}'.format(new_spacing, image.shape))


    return image, new_spacing

def load_mhd(filepath):
    itk_img = sitk.ReadImage(filepath)
    # img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
    # num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
    # origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    # spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
    return itk_img

#-
def file_to_array(filepath, newSpacing=(1, 1, 1), verbose=False):
    files_dicom = glob.glob(filepath + '/*.dcm')
    # files_mhd = glob.glob(filepath + '/*.mhd')
    if files_dicom:
        rawDataAry = load_scan(filepath)
        patient_pixels = get_pixels_hu(rawDataAry)
        pix_resampled, spacing = resample_dcm(patient_pixels, rawDataAry, newSpacing, verbose=verbose)
    # return pix_resampled, spacing
    # print(filename.split('.'))
    elif '.mhd' in filepath:
        rawDataAry = load_mhd(filepath)

        # patient_pixels = np.swapaxes(patient_pixels, 2, 0)
        origin = np.array(rawDataAry.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        pix_resampled, spacing = resample_mhd(rawDataAry, newSpacing, verbose=verbose)
    else:
        return None, None
    return pix_resampled, spacing


def process_downsample(subset):
    drive = 'tera'
    wd = '/media/mike/{}/data/bowl17/'.format(drive)
    # input_folder = wd+'/kgsamples/' # '/CTscans/'
    input_folder = wd + '/luna/subset{}/'.format(subset)
    resampled_folder = wd + '/resampled_images/'
    downsampled_folder = wd + '/downsampled_images/'

    if not os.path.exists(resampled_folder):
        os.makedirs(resampled_folder)

    if not os.path.exists(downsampled_folder):
        os.makedirs(downsampled_folder)

    patients = os.listdir(input_folder)
    patients.sort()

    t1 = time.time()
    print(input_folder)
    print('# of samples: ', len(patients))
    for i in tqdm(range(0, len(patients))):
        # patient = load_scan(input_folder + patients[i])
        # patient_pixels = get_pixels_hu(patient)
        # pix_resampled, spacing = resample(patient_pixels, patient, [1,1,1])
        if not os.path.exists(resampled_folder + patients[i] + '.npy'):
            pix_resampled, spacing = file_to_array(input_folder + patients[i], verbose=True)
            if pix_resampled is not None:
                np.save(resampled_folder + patients[i], pix_resampled)

                # ratio = 0.5
                # downsampled = scipy.ndimage.interpolation.zoom(pix_resampled, [ratio, ratio, ratio])
                # np.save(downsampled_folder + patients[i] + "_.5", downsampled)
                #
                #
                # ratio = 0.25
                # downsampled = scipy.ndimage.interpolation.zoom(pix_resampled, [ratio, ratio, ratio])
                # np.save(downsampled_folder + patients[i] + "_.25", downsampled)

                # print("done with %1i" % i)
        else:
            print('skip')

#-----------------------------------------------------

if __name__ == '__main__':
    # wd = '/gpfs/scratch/delton17'
    # try:
    #     ss = int(sys.argv[1])
    # except:
    #     raise ValueError('Invalid command line argument specified')

    for i in [0,1]:
        process_downsample(i)



