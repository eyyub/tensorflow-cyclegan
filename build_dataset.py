import os
import sys
import numpy as np
from PIL import Image, ImageOps

def _image_preprocessing(filename, xsize, ysize):
    im = Image.open(filename)

    if im.mode != 'RGB':
        print('Mode: ', im.mode)
        tmp = im.convert('RGB')
        im.close()
        im = tmp

    downsampled_im = ImageOps.fit(im, (xsize, ysize), method=Image.LANCZOS)
    norm_im = np.array(downsampled_im, dtype=np.float32)

    downsampled_im.close()
    im.close()
    return norm_im

if __name__ == '__main__':
    pathA = sys.argv[1]
    pathB = sys.argv[2]
    namesA = []
    namesB = []

    for name in os.listdir(pathA):
        namesA.append(os.path.join(pathA, name))

    for name in os.listdir(pathB):
        namesB.append(os.path.join(pathB, name))

    dataset_A = np.zeros((len(namesA), 256, 256, 3))
    dataset_B = np.zeros((len(namesB), 256, 256, 3))

    for i in range(len(namesA)):
        dataset_A[i] = _image_preprocessing(namesA[i], 256, 256)
        print(namesA[i])

    for i in range(len(namesB)):
        dataset_B[i] = _image_preprocessing(namesB[i], 256, 256)
        print(namesB[i])

    np.save('dataset_%s.npy' % sys.argv[3], dataset_A)
    np.save('dataset_%s.npy' % sys.argv[4], dataset_B)
