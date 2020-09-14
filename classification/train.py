from sklearn.model_selection import train_test_split 
import torch.optim as optim
from prepare_data import categories
from prepare_data import datasets 
from preprocess import word2index
from preprocess import  sentence2index
from model import LSTMClassifier
from model import category2tensor
import torch.nn as nn

# 元データを7:3に分ける
train_data, test_data = train_test_split(datasets, train_size=0.7)

# 単語のベクトル次元数
EMBEDDING_DIM = 10
# 隠れ層の次元数
HIDDEN_DIM = 128
# データ全体の単語数
VOCAB_SIZE = len(word2index)
# 分類先のカテゴリの数
TAG_SIZE = len(categories)
# モデル宣言
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
# 損失関数はNLLLoss()。LogSoftmaxにはこれを使うことが多い。
loss_function = nn.NLLLoss()
# 最適化はSGD。lossの減りに多少時間がかかる。
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 各エポックの合計loss値を格納
losses = []

for epoch in range(100):
    all_loss = 0
    for title, cat in zip(train_data["title"], train_data["category"]):
        # モデルが持っている勾配の情報をリセット
        model.zero_grad()
        # 文章を単語IDの系列に変換（modelが読み込めるように）
        inputs = sentence2index(title)
        # 順伝播
        out = model(inputs)
        # 正解カテゴリをテンソル化
        answer = category2tensor(cat)
        # 正解とのlossを計算
        loss = loss_function(out, answer)
        # 勾配をセット
        loss.backward()
        # 逆伝播でパラメータ更新
        optimizer.step()
        # lossを集計
        all_loss += loss.item()
    losses.append(all_loss)
    print("epoch", epoch, "\t", "loss", all_loss)
print("done.")