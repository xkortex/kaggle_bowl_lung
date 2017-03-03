import numpy as np, pandas as pd
import glob
import os, sys
from tqdm import tqdm

# class Get_Npz_Shape():
#     def __init__(self, filename, verbose=False):
#         self.filename = filename
#
#     def __enter__(self):
#         self.start = time.time()
#         return self
#
#     def __exit__(self, *args):
#         self.end = time.time()
#         self.sec = self.end - self.start
#         self.msec = self.sec * 1000  # millisecs
#         if self.verbose:
#             self._print_formatted()

def get_npz_shape(filename):
    myfile = np.load(resampled_folder + patients[i] + '.npy')
    return list(np.shape(myfile))

if __name__ == '__main__':
    # wd = '/gpfs/scratch/delton17'
    # input_folder = wd + '/CTscans/'
    wd = '/media/mike/tera/data/databowl/'
    input_folder = wd + '/kgsamples/' # '/CTscans/'
    resampled_folder = wd + '/resampled_images/'
    # downsampled_folder = wd + '/downsampled_images/'

    patients = os.listdir(input_folder)
    patients.sort()
    np.array()

    shapes = []
    for i in tqdm(range(0, len(patients))):
        try:
            shape = get_npz_shape(resampled_folder + patients[i] + '.npy')
            row = [patients[i], ] + shape
            shapes.append(row)
        except KeyboardInterrupt:
            sys.exit()
        except Exception as exc:
            print(exc)
        if i % 5 == 0:
            shapes_df = pd.DataFrame(shapes, columns=['id', 'z', 'y', 'x'])
            shapes_df.to_csv(wd + '/' + 'shapes.csv', index=False)
    shapes_df = pd.DataFrame(shapes, columns=['id','z', 'y', 'x'])
    shapes_df.to_csv(wd + '/' + 'shapes.csv', index=False)


