import numpy as np
import torch  
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from torchvision import transforms, models

from tqdm import tqdm
from sklearn.base import BaseEstimator

class SuperResolutionNet(nn.Module):
    def __init__(self, n_channels):
        super(SuperResolutionNet, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=4, stride=2,padding=1             
        )

        
    def forward(self, x):
        ## x.shape = [B, 64, 64]
        x = x.unsqueeze(1)
        x = self.upconv(x) # [B, 1, 128, 128]
        x = x.view(-1, 128*128) # [B, 128*128]
    
        return x ## [B, 128 * 128]
        

class Regressor(BaseEstimator):
    def __init__(self, n_channels=1, n_epochs=10, batch_size=8, lr=0.001):
        self.n_channels = n_channels 
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        ## X.shape = [B, 64, 64]
        ## y.shape = [B, 128 * 128]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        self.model = SuperResolutionNet(self.n_channels)
        self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        ## convert Y to grayscale (Not necessary if Y is already grayscale)
        # transform_Y = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        # y = transform_Y(y.permute(0,3,1,2)).view(-1, 128, 128) ## [B, 128, 128]
        
        ## select only the first channel on X
        # X = X[:, :, :, 0]
            
        trainset = TensorDataset(X, y)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2) 
        for epoch in range(self.n_epochs):
            running_loss = 0.0
            for data in trainloader:
                inputs, labels = data ## inputs.shape = [B, 64, 64], labels.shape = [B, 128*128]
                inputs, labels = inputs.to(self.device), labels.to(self.device) 
                optimizer.zero_grad()
                outputs = self.model(inputs) ## [B, 128*128]
                # print(outputs.shape)
                # print(labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            #print(f'Epoch {epoch+1}/{self.n_epochs}, Loss: {running_loss/len(trainloader)}')

    def predict(self, X):
        ## X.shape = [B, 64, 64]
        X = torch.tensor(X, dtype=torch.float32)
        # X = X[:, :, :, 0]
        
        testloader = DataLoader(X, batch_size=self.batch_size, shuffle=False, num_workers=2)
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for images in testloader:
                images = images.to(self.device)
                outputs = self.model(images) ## [B, 128*128]
                y_pred.extend(outputs.cpu().numpy()) ## [B, 128*128]
        return np.array(y_pred).reshape(X.shape[0], -1) ## [test_size, 128*128]
    

if __name__ == '__main__':
    # Check that everything works fine
    import sys
    import os
    
    ## -- Test with the real data --
    script_dir = os.path.dirname(__file__)
    project_dir = os.path.join(script_dir, '..', '..')
    print(project_dir)
    
    sys.path.append(project_dir)
    from problem import get_train_data, get_test_data
    
    X, y = get_train_data(path = os.path.join(project_dir, 'data', 'public'))
    X_test, y_test = get_test_data(path = os.path.join(project_dir, 'data', 'public'))
    
    print(X.shape, y.shape)
    print(X_test.shape, y_test.shape)
    ## max and min values of X and y
    print(X.max(), X.min())
    print(y.max(), y.min())
    print(X_test.max(), X_test.min())
    
    ## -- Quick test --
    # X, y = np.random.rand(100, 64, 64), np.random.rand(100, 320*320)
    # X_test = np.random.rand(100, 64, 64)
    
    
    clf = Regressor()
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    print(y_pred.shape)
    
    
    # clf = Regressor()
    # clf.fit(X, y)
    # y_pred = clf.predict(X_test)
    # print(y_pred.shape)
