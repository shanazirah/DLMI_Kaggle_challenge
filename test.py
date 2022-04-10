import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from tqdm.notebook import tqdm
import pandas as pd
import zipfile
import os
import cv2

from tile_dataset import TileDataset
from image_preprocessing.py import create_patch_image

TEST_DIR = '/content/test/test_tiles'
TEST_FILES = '/content/test.csv'
TEST_IMAGE_PATH = '/content/test/test/'
device = torch.device('cuda')

def test(model, device, data_loader):
    model.eval()
    Y_pred_all = torch.empty(0)
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data = torch.squeeze(data[0].cuda())

            Y_prob = model(data)
            y_pred_softmax = torch.log_softmax(Y_prob, dim=1)
            _, Y_pred = torch.max(y_pred_softmax, dim=1)

            Y_pred_all = torch.cat((Y_pred_all, Y_pred.detach().cpu()))

    return Y_pred_all

if __name__ == '__main__':

    OUT_TEST = '/content/test_tiles.zip'
    names = [f for f in os.listdir(TEST_IMAGE_PATH)]

    with zipfile.ZipFile(OUT_TEST, 'w') as img_out:
    for name in tqdm(names):
        image_path = os.path.join(TEST_IMAGE_PATH, name)
        result = create_patch_image(image_path, "")

        for r in result:
                img, idx = r['img'], r['idx']
                img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                img_out.writestr(f'{name[:-5]}_{idx}.png', img)
    
    test_df = pd.read_csv(TEST_FILES)
    transform_test = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomVerticalFlip(0.5),
                                        transforms.ToTensor()])

    test_set = TileDataset(TEST_DIR, test_df, mode = 'test', transform=transform_test)

    test_loader = data_utils.DataLoader(test_set, 1, shuffle=True, num_workers=0)

    test_df['predicted'] = test(model, device, test_loader)
    test_df['predicted'] = test_df['predicted'].astype('int64')
    test_df.head()