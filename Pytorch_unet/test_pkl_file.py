import os
import pickle
import scipy.io as sio

MG_CHANNELS = 1
IMG_HEIGHT= 256                                                                                    
IMG_WIDTH = 512
WEIGHTS_PATH = './weights/weights_train1.pth'
TEST_PATH = './Test_data/'
SAVE_PATH = './Results/' + (WEIGHTS_PATH.split('_')[1].split('.')[0])
lst = os.listdir(TEST_PATH)
n_samples = len(lst)//2

full = 'original'
limited = 'limited_noise_interpolated'
files = os.listdir(TEST_PATH)

for file in files:
    if (file.startswith('original')):
        continue
    load_path = TEST_PATH + file
    # Load image and add reflective padding
    print(file)
    mat = sio.loadmat(load_path)
    test_img = np.asarray(mat[limited], dtype='float32')
    test_img = np.pad(test_img, [(28,28),(0,0) ], mode = 'reflect')
    l.

with open('./data/Test.pkl', 'wb') as f:
	pickle.dump(l, f)
