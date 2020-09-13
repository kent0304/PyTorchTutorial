from preprocess import  sentence2index
from preprocess import word2index
import torch.nn as nn

# 全単語数を取得
VOCAB_SIZE = len(word2index)
# 単語のベクトル数
EMBEDDING_DIM = 10
# 隠れ層
HIDDEN_DIM = 128
embeds = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)

sentence = "震災をうけて感じた、大切だと思ったこと"
input = sentence2index(sentence)
emb = embeds(input)
print(emb)
# バッチを取り入れるために変換
lstm_input = emb.view(len(input),1,-1)
out1, out2 = lstm(lstm_input)


# nn.Moduleを継承してクラス定義
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # 単語のベクトル化
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # lstmの隠れ層は以下のみでいい
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # LSTMの出力をsoftmaxに食わせるために変換する一層
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        # softmaxのlogバージョン。dim=0で列、dim=1で行方向を確率変換
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence):
        # 文章内の各単語をベクトル化して出力
        embeds = self.word_embeddings(sentence)
        # 2次元テンソルをLSTMをviewで3次元テンソルに変換
        # 今回は many to oneのタスクを使うので、第二戻り値のみ利用
        _, lstm_out = self.lstm(embeds.view((len(sentence), 1, -1)))
        # lstm_out[0]は3次元テンソルなので2次元に調整
        tag_space = self.hidden2tag(lstm_out[0].view(-1, HIDDEN_DIM))
        # softmaxを用いて確率として表現
        tag_score = self.softmax(tag_score)
        return tag_scores
        

