import openslide
from openslide import OpenSlideError
import numpy as np
import zipfile
from tqdm.notebook import tqdm
import cv2
import os

N_TILES = 64
TILE_SIZE = 64
IMG_SIZE = 2056
TRAIN_IMAGE_PATH = '/content/train/train/'
TRAIN_MASKS_PATH = '/content/train_label_masks/train_label_masks/'
OUT_TRAIN = '/content/train_tiles.zip'
OUT_MASKS = '/content/masks_tiles.zip'

def create_patch_image(image_path: str):
    """
        Divide the image into N_TILES images of size IMG_SIZE
    """
    img_slide = openslide.OpenSlide(image_path)
    img = np.array(img_slide.get_thumbnail((IMG_SIZE, IMG_SIZE)))

    result = []
    shape = img.shape

    pad0, pad1 = (TILE_SIZE - shape[0]%TILE_SIZE)%TILE_SIZE, (TILE_SIZE - shape[1]%TILE_SIZE)%TILE_SIZE
    img = np.pad(img, [[pad0//2, pad0 - pad0//2],[pad1//2, pad1 - pad1//2],[0,0]],
                        constant_values=255)
    img = img.reshape(img.shape[0]//TILE_SIZE, TILE_SIZE, img.shape[1]//TILE_SIZE, TILE_SIZE, 3)
    img = img.transpose(0,2,1,3,4).reshape(-1, TILE_SIZE, TILE_SIZE, 3)
    
    if len(img) < N_TILES:
        img = np.pad(img,[[0,N_TILES-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

    idxs = np.argsort(img.reshape(img[:,:,0].shape[0],-1).sum(-1))[:N_TILES]
    img = img[idxs]


    for i in range(len(img)):
        result.append({'img':img[i], 'idx':i})

    return result

if __name__ == '__main__':
    names = [f for f in os.listdir(TRAIN_MASKS_PATH)]
    i=0

    with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out,\
    zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
        for name in tqdm(names):
            image_path = os.path.join(TRAIN_IMAGE_PATH, name)
            result = create_patch_image(image_path)

            for r in result:
                img, idx = r['img'], r['idx']
                img = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr(f'{name[:-5]}_{idx}.png', img)