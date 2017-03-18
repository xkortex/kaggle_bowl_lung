import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import glob



import SimpleITK as sitk


def get_origin(filepath):
    """
    Get the origin coordinates from a ITK file.
    ITK files contain a parameter which references the origin of the reference frame system.
    :param filepath: path to a .mhd file
    :return: Origin coordinate
    """
    itk_img = sitk.ReadImage(filepath)
    origin = np.array(itk_img.GetOrigin())
    return origin


def get_filename(uid, file_list=None):
    """
    Return the absolute path to a file, given a patient UID
    :param uid: Paitent UID
    :param file_list: List of files with .mhd files
    :return: a filename
    """
    if file_list is None:
        raise ValueError('You must specify a list of file paths as a keyword argument to Pandas df.apply(), e.g.:\ '
                         '  df_node["file"] = df_node["seriesuid"].apply(get_filename, file_list=mhd_file_list)')
    for f in file_list:
        if uid in f:
            return(f)
    return None # this will later be dropped from the dataframe

def strip_uid(path):
    """Helper to convert path to UID"""
    fname = os.path.basename(path)
    return fname.strip('.mhd.npy')


def load_node_df(annotations_path, luna_subset_path):
    """
    Load in the node annotations file, and link it up with the paths to the actual files.
    :param annotations_path:
    :param luna_subset_path:
    :return:
    """
    df_node = pd.read_csv(annotations_path)
    mhd_file_list = glob.glob(luna_subset_path + "*.mhd")
    df_node["file"] = df_node["seriesuid"].apply(get_filename, file_list=mhd_file_list)
    df_node = df_node.dropna()
    return df_node