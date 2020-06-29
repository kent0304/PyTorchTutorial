from sklearn import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
iris = datasets.load_iris()
# 教師ラベルをダミー変数化する必要はない
# y = np.zeros((len(iris.target), 1+iris.target.max()), dtype=int)
# y[np.arange(len(iris.target)), iris.target] = 1
y = iris.target



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.2, random_state=0)

X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

### Training
# ネットワークのインスタンス化
net = Net()
# パラメタータ更新手法、学習率の指定
optimizer = optim.SGD(net.parameters(), lr=0.001)
# 目的関数の指定
criterion = nn.CrossEntropyLoss()

for i in range(3000):
    # 古い勾配は削除
    optimizer.zero_grad()
    output = net(X_train)
    loss = criterion(output, y_train)
    # バックプロパゲーションを用いて目的関数の微分を計算
    loss.backward()
    optimizer.step()


### Prediction
outputs = net(torch.tensor(X_test, dtype=torch.float))
_, predicted = torch.max(outputs.data, 1)

# print(outputs.data)
# print(torch.max(outputs.data, 1))
# 2つめの引数に1をつけることで、最大値とそのインデックスを順に出力

print(y_test)
accuracy = 100 * np.sum(predicted.numpy() == y_test) / len(iris.target)
print('accuracy = {:.1f}%'.format(accuracy))
