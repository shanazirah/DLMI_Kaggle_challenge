import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import math
import PIL
from PIL import Image

N_TILES = 64

class TileDataset(data_utils.Dataset):
    """
    Custom dataset class for tiles
    __getitem__ returns a batch of N_TILES tiles along with their isup_grade
    """
    def __init__(self, img_path, df, mode='train', transform=None):
        """
        img_zip: Where the images are stored
        df: The train.csv dataframe
        mode: either train/test
        transform: The function to apply to the image. Usually data augmentation.
        """
        self.img_path = img_path
        self.df = df.reset_index(drop=True)
        self.img_list = list(self.df['image_id'])
        self.mode = mode
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.img_list[idx]
        
        tiles = ['/' + img_id + '_' + str(i) + '.png' for i in range(0, N_TILES)]
    
        image_tiles = []

        for tile in tiles:
          image = Image.open(self.img_path + tile)
          
          if self.transform:
            image = self.transform(image)
          
          image = 1 - image
          image_tiles.append(image)

        image_tiles = torch.stack(image_tiles, dim=0)
        
        if self.mode == 'train':
          isup_grade = self.df.loc[idx, 'isup_grade']
          return torch.tensor(image_tiles), torch.tensor(isup_grade)
        else:
          return torch.tensor(image_tiles)

    def __len__(self):
        return len(self.img_list)