from sklearn.model_selection import train_test_split 
import torch.optim as optim
from prepare_data import categories
from prepare_data import datasets 
from preprocess import word2index
from preprocess import  sentence2index
from model import LSTMClassifier
from model import category2tensor
from model import category2index
import collections 
import torch.nn as nn
import pandas as pd 

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



# IDをカテゴリに戻す用
index2category = {}
for cat, idx in category2index:
    index2category[idx] = cat 

# answerは正解ラベル、predictはLSTMの予測結果、exactは正解していればO間違ってればX
predict_df = pd.DataFrame(columns=["answer", "predict", "exact"])

# テストデータの母数計算
test_num = len(test_data)
# 正解の件数
ans = 0
# 勾配自動計算OFF
with torch.no_grad():
    for title, category in zip(test_data["title"], test_data["category"]):
        # テストデータの予測
        inputs = sentence2index(title)
        out = model(inputs)

        # outの一番大きい要素が予測結果
        _, predict = torch.max(out, 1)
        answer = category2tensor(category)
        exact = "O" if predict.item() == answer.item() else "X"
        s = pd.Series([answer.item(), predict.item(), exact], index=predict_df.columns)
        predict_df = predict_df.append(s, ignore_index=True)

# Fスコア格納用のDF
fscore_df = pd.DataFrame(columns=["category", "all", "precision", "recall", "fscore"])

# 分類器が答えた各カテゴリの件数
prediction_count = collections.Counter(predict_df["predict"])
# 各カテゴリの総件数
answer_count = collections.Counter(predict_df["answer"])

# Fスコアを求める
for i in range(9):
    all_count = answer_count[i]
    precision = len(predict_df.query('predict == ' + str(i) + ' and exact == "O"')) / prediction_count[i]
    recall = len(predict_df.query('answer == ' + str(i) + ' and exact == "0"')) / all_count 
    fscore = 2*precision*recall / (precision + recall)
    s = pd.Series([index2category[i], all_count, round(precision, 2), round(recall, 2), round(fscore, 2)], index=fscore_df.columns)
    fscore_df = fscore_df.append(s, ignore_index=True)
print(fscore_df)