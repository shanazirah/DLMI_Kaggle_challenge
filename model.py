import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import pandas as pd
from tile_dataset import TileDataset

N_TILES = 64
TRAIN_FILES = '/content/train.csv'
TRAIN_DIR = '/content/train/train_tiles'
n_epochs = 50
lr=0.0001
betas=(0.9, 0.999)
weight_decay=0.0005

class MILAttentionModel(nn.Module):
    def __init__(self):
        super(MILAttentionModel, self).__init__()
        self.L = 512 # node fully connected layer
        self.D = 128 # node attention layer
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
         
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(N_TILES * 48 * 14 * 14, self.L), 
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 6),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, N_TILES * 48 * 14 * 14)
        H = self.feature_extractor_part2(H)

        A = self.attention(H) 
        A = torch.transpose(A, 1, 0) 
        A = F.softmax(A, dim=1)

        M = torch.mm(A, H)

        Y_prob = self.classifier(M)
        

        return Y_prob

def multi_acc(y_pred, y_test):
  y_pred_softmax = torch.log_softmax(y_pred, dim=1)
  _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
  correct_pred = (y_pred_tags == y_test).float()
  acc = correct_pred.sum() / len(correct_pred)
  acc = torch.round(acc * 100)

  return y_pred_tags, acc

def train(model, device, train_loader, criterion, optimizer, epoch):
    train_loss = 0.
    y_pred_all=torch.empty(0,6)
    label_all=torch.empty(0)
    bar = tqdm(train_loader)
    for data in bar:
        img, label = data[0].cuda(), data[1].cuda()
        label = label.long()
        img = torch.squeeze(img)

        # Reset gradients
        optimizer.zero_grad()

        # Keep track of predictions and labels to calculate accuracy after each epoch
        Y_prob = model(img)
        
        loss = criterion(Y_prob, label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        y_pred_all = torch.cat((y_pred_all, Y_prob.detach().cpu()))
        label_all = torch.cat((label_all, label.detach().cpu()))


    # Calculate loss and error for epoch
    train_loss /= len(train_loader)
    print('Train Set, Epoch: {}, Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, 
                                                                         train_loss, 
                                                                         multi_acc(y_pred_all, label_all)[1]))
    return train_loss

if __name__ == '__main__':
    train_df = pd.read_csv(TRAIN_FILES)

    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomVerticalFlip(0.5),
                                        transforms.ToTensor()])

    train_set = TileDataset(TRAIN_DIR, train_df, mode='train', transform=transform_train)
    train_loader = data_utils.DataLoader(train_set, 1, shuffle=True, num_workers=0)
    device = torch.device('cuda')
    model = MILAttentionModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
    loss = []
    for epoch in range(1, n_epochs):
            loss.append(train(model, device, train_loader, criterion, optimizer, epoch))