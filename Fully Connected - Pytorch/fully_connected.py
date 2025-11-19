import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import numpy as np


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



class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 50)  
        self.fc2 = nn.Linear(50, 26)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)               
        return x
    


dataset = A_Z_Dataset()
# print("len: ", len(dataset))
# print("item: ", dataset[2])

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
# for img, labl in dataloader:
#     print("img: ", img.shape)
#     print("lbl: ", labl.shape)
#     break


model = SimpleNN()
model.to(device)
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss() 
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(optimizer)

for epoch in range(4):
    for inputs, labels in dataloader:

        inputs, labels = inputs.to(device), labels.to(device)

        model.train()

        outputs = model(inputs)

        # print("outputs: ", outputs[0])
        # print("labels: ", labels[0])

        loss = criterion(outputs, labels)
        # print("loss: ", loss)
        # input()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    test_data_cpu = dataset.X_test
    Label_cpu = dataset.y_test
    test_data = test_data_cpu.to(device)
    Label = Label_cpu.to(device)
    predictions = model(test_data)
    val, predictions = torch.max(predictions, dim=1)
    # print("Prediction: ", predictions[0:10], "  Label: ", Label[0:10])
    # print("Shapes: ", predictions.shape, "  ", Label.shape)

    temp = predictions - Label
    correct = len(temp) - torch.count_nonzero(temp)
    acc = correct / len(temp)
    print("Accuracy: ", acc)

    plt.figure()
    R = torch.round(torch.rand(1) * (len(temp)-1)).int()
    # print("R: ", R)
    img = test_data_cpu[R]
    img = img.resize(28, 28)
    # print("img: ", img)
    print("Predicted as: ", Alphabet[Label_cpu[R].item()])
    plt.title("Random sample from the test set.")
    plt.imshow(np.round(img.numpy()*255), cmap="gray")
    # plt.colorbar()
    plt.show()

