import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import numpy as np
# import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns


Alphabet = {0: 'A',
            1: 'B', 
            2: 'C', 
            3: 'D', 
            4: 'E', 
            5: 'F', 
            6: 'G', 
            7: 'H', 
            8: 'I', 
            9: 'J', 
            10: 'K',
            11: 'L', 
            12: 'M', 
            13: 'N', 
            14: 'O', 
            15: 'P', 
            16: 'Q', 
            17: 'R', 
            18: 'S', 
            19: 'T', 
            20: 'U', 
            21: 'V', 
            22: 'W', 
            23: 'X', 
            24: 'Y', 
            25: 'Z'}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class A_Z_Dataset(Dataset):
    def __init__(self, transform=None):
        super(A_Z_Dataset, self).__init__()
        df = pd.read_csv("C:/Users/Lenovo/Projects/Datasets/A_Z Handwritten/A_Z Handwritten Data.csv")

        X = torch.tensor(df.values, dtype=torch.float32)

        Y = X[:, 0].long() 
        X = X[:, 1:] / 255.0 

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

        self.X_train = self.X_train.reshape([self.X_train.shape[0], 28, 28])
        self.X_test = self.X_test.reshape([self.X_test.shape[0], 28, 28])

        print("X train: ", self.X_train.shape)
        print("y train: ", self.y_train.shape)
        print("X test: ", self.X_test.shape)
        print("y test: ", self.y_test.shape)

        # scaler = StandardScaler()
        # self.X_train = scaler.fit_transform(self.X_train)
        # self.X_test = scaler.transform(self.X_test)

    def __len__(self):
        return len(self.X_train)
    

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)   # 5 channel 24*24
        self.pool = nn.MaxPool2d(2, 2)    # 5 channel 12*12
        self.conv2 = nn.Conv2d(5, 10, 5)  # 10 channel 8*8 --> 10 channel 4*4
        self.fc1 = nn.Linear(160, 10)
        self.fc2 = nn.Linear(10, 26)
        # n*n, k*k, (n-k+1)*(n-k+1)

    def forward(self, x):
        x = x.unsqueeze(1)
        # print("before: ", x.shape)
        x = self.pool(torch.relu(self.conv1(x)))
        # print("conv1: ", x.shape)
        x = self.pool(torch.relu(self.conv2(x)))
        # print("conv2: ", x.shape)
        x = x.view(x.size(0), -1)
        # print("flatten: ", x.shape)
        x = torch.relu(self.fc1(x))
        # print("fc1: ", x.shape)
        x = self.fc2(x)      
        # print("fc2: ", x.shape)       
        # input('hey') 
        return x
    


dataset = A_Z_Dataset()
# print("len: ", len(dataset))
# print("item: ", dataset[2])

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
# for img, labl in dataloader:
#     print("img: ", img.shape)
#     print("lbl: ", labl.shape)
#     break


model = Net()
model.to(device)
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss() 
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(optimizer)

for epoch in range(15):
    L = 0
    for inputs, labels in dataloader:

        inputs, labels = inputs.to(device), labels.to(device)

        model.train()

        outputs = model(inputs)

        # print("outputs: ", outputs[0])
        # print("labels: ", labels[0])

        loss = criterion(outputs, labels)
        # print("loss: ", loss)
        # input()
        L += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch number: ", epoch+1, "      Loss: ", L)


model.eval()
with torch.no_grad():
    test_data_cpu = dataset.X_test
    Label_cpu = dataset.y_test
    test_data = test_data_cpu.to(device)
    Label = Label_cpu.to(device)
    predictions = model(test_data)
    val, predictions = torch.max(predictions, dim=1)
    predictions_cpu = predictions.cpu()
    # print("Prediction: ", predictions[0:10], "  Label: ", Label[0:10])
    # print("Shapes: ", predictions.shape, "  ", Label.shape)

    temp = predictions - Label
    correct = len(temp) - torch.count_nonzero(temp)
    acc = correct / len(temp)
    print("Accuracy: ", acc)


    cm = confusion_matrix(Label_cpu.numpy(), predictions_cpu.numpy())
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(Alphabet.values()), yticklabels=list(Alphabet.values()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


    plt.figure()
    R = torch.round(torch.rand(1) * (len(temp)-1)).int()
    # print("R: ", R)
    img = test_data_cpu[R]
    img = img.resize(28, 28)
    # print("img: ", img)
    print("Predicted as: ", Alphabet[predictions_cpu[R].item()])
    plt.title("Random sample from the test set.")
    plt.imshow(np.round(img.numpy()*255), cmap="gray")
    # plt.colorbar()
    plt.show()

