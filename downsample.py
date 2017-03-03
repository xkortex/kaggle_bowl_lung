#----------------------------------------
#--- Downsampling routine 
#--- Dan Elton 
#----------------------------------------

import numpy as np 
import dicom
import SimpleITK as sitk
import os
import scipy.ndimage
import glob

import time 
from tqdm import tqdm

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
def resample_dcm(image, scan, new_spacing=[1, 1, 1]):
    '''Resampling to make x-y-z pixel spacing standardized'''
    
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing


def resample_mhd(image, new_spacing=[1, 1, 1]):
    '''Resampling to make x-y-z pixel spacing standardized'''

    # Determine current pixel spacing
    spacing = np.array(image.GetSpacing())  # spacing of voxels in world coor. (mm)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def load_mhd(filepath):
    itk_img = sitk.ReadImage(filepath)
    # img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
    # num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
    # origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    # spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
    return itk_img

#-
def file_to_array(filepath, newSpacing=(1, 1, 1)):
    files_dicom = glob.glob(filepath + '/*.dcm')
    files_mhd = glob.glob(filepath + '/*.mhd')
    if files_dicom:
        rawDataAry = load_scan(filepath)
        patient_pixels = get_pixels_hu(rawDataAry)
        pix_resampled, spacing = resample_dcm(patient_pixels, rawDataAry, newSpacing)
    # return pix_resampled, spacing
    # print(filename.split('.'))
    elif files_mhd:
        rawDataAry = load_mhd(filepath)
        patient_pixels = sitk.GetArrayFromImage(rawDataAry)
        origin = np.array(rawDataAry.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        pix_resampled, spacing = resample_mhd(patient_pixels, newSpacing)
    else:
        return None, None
    return pix_resampled, spacing



#-----------------------------------------------------
# wd = '/gpfs/scratch/delton17'
wd = '/media/mike/tera/data/databowl/'
input_folder = wd+'/kgsamples/' # '/CTscans/'
resampled_folder = wd + '/resampled_images/'
downsampled_folder = wd + '/downsampled_images/'

if not os.path.exists(resampled_folder):
    os.makedirs(resampled_folder)
    
if not os.path.exists(downsampled_folder):
    os.makedirs(downsampled_folder)

patients = os.listdir(input_folder)
patients.sort()

t1 = time.time()
print(patients)
for i in tqdm(range(0,len(patients))):
    # patient = load_scan(input_folder + patients[i])
    # patient_pixels = get_pixels_hu(patient)
    # pix_resampled, spacing = resample(patient_pixels, patient, [1,1,1])
    pix_resampled, spacing = file_to_array(input_folder + patients[i])
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

