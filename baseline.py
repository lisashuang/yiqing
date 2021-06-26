import pandas as pd
import numpy as np
import jieba
import time
import re
import torch
import torch.nn as nn
from torchtext import data
from torchtext.data import Field
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

train_df = pd.read_csv(r'data\nCoV_100k_train.labled_utf8.csv', encoding='utf-8', error_bad_lines=False)
test= pd.read_csv(r'data\nCov_10k_test.csv', encoding='utf-8')
train_df_copy = train_df.copy()
test_copy = test.copy()
#检查一下训练集和测试集的缺失值，可以看出有些数据是缺失的
print(train_df.isnull().any())
print(test.isnull().any())
#删除包含缺失值的行
train_df.dropna(inplace=True)
test.dropna(inplace=True)

#分析数据
#发微博数量与时间关系

train_df_copy['time'] = pd.to_datetime('2020年' + train_df['微博发布时间'], format='%Y年%m月%d日 %H:%M', errors='ignore')
test_copy['time'] = pd.to_datetime('2020年' + train_df['微博发布时间'], format='%Y年%m月%d日 %H:%M', errors='ignore')
train_df_copy['month'] =  train_df_copy['time'].dt.month
train_df_copy['day'] =  train_df_copy['time'].dt.day
train_df_copy['dayfromzero']  = (train_df_copy['month'] - 1) * 31 +  train_df_copy['day']
test_copy['month'] =  test_copy['time'].dt.month
test_copy['day'] =  test_copy['time'].dt.day
test_copy['dayfromzero']  = (test_copy['month'] - 1) * 31 +  test_copy['day']
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.kdeplot(train_df_copy.loc[train_df_copy['情感倾向'] == '0', 'dayfromzero'], ax=ax[0], label='sent(0)')
sns.kdeplot(train_df_copy.loc[train_df_copy['情感倾向'] == '1', 'dayfromzero'], ax=ax[0], label='sent(1)')
sns.kdeplot(train_df_copy.loc[train_df_copy['情感倾向'] == '-1', 'dayfromzero'], ax=ax[0], label='sent(-1)')
train_df_copy.loc[train_df_copy['情感倾向'] == '0', 'dayfromzero'].hist(ax=ax[1])
train_df_copy.loc[train_df_copy['情感倾向'] == '1', 'dayfromzero'].hist(ax=ax[1])
train_df_copy.loc[train_df_copy['情感倾向'] == '-1', 'dayfromzero'].hist(ax=ax[1])
ax[1].legend(['sent(0)', 'sent(1)','sent(-1)'])
plt.show()

#现在开始统计相关微博的长度，训练集和测试集都有
train_df_copy['Chinese_Content_Length'] = train_df['微博中文内容'].astype(str).apply(len)
test_copy['Chinese_Content_Length'] = train_df['微博中文内容'].astype(str).apply(len)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,8))
sns.distplot(train_df_copy['Chinese_Content_Length'], ax=ax1, color='blue')
sns.distplot(test_copy['Chinese_Content_Length'], ax=ax2, color='green')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax1.set_title('训练集微博长度分布')
ax2.set_title('测试集微博长度分布')
plt.show()

#采用 jieba 和 wordcloud 对正文做一个词云。
stop = open('data/stopwords.txt', 'r+', encoding='utf-8')
stopword = stop.read().split("\n")
stopeword = set(stopword)
stop.close()

def stripword(seg):
    """停用词处理"""
    wordlist = []
    for key in seg.split(' '):
        #去除停用词和单字
        if not (key.strip() in stopword) and (len(key.strip()) > 1):
            wordlist.append(key)
    return ' '.join(wordlist)
def cutword(content):
    """分词，去除停用词，写得比较简陋"""
    seg_list = jieba.cut(content)
    line = " ".join(seg_list)
    word = stripword(line)
    return word

train_df_copy['Chinese_Content_cut'] = train_df['微博中文内容'].astype(str).apply(cutword)
font = r'C:\\Windows\\fonts\\msyh.ttc' #字体
wc = WordCloud(font_path=font,
               max_words=2000,
               width=1800,
               height=1600,
               mode='RGBA',
               background_color=None).generate(str(train_df_copy['Chinese_Content_cut'].values))
plt.figure(figsize=(14, 12))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

#图片统计
train_df_copy['Pic_Length'] = train_df_copy['微博图片'].apply(lambda x: len(eval(x)))
test_copy['Pic_Length'] = test_copy['微博图片'].apply(lambda x: len(eval(x)))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
ax1.set_xlim(0, 9)
ax2.set_xlim(0, 9)
sns.distplot(train_df_copy['Pic_Length'], bins=25, ax=ax1, color='blue', kde=False)
sns.distplot(test_copy['Pic_Length'], bins=25, ax=ax2, color='green', kde=False)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax1.set_title('训练集图片数量分布')
ax2.set_title('测试集图片数量分布')
plt.show()

#视频统计
train_df_copy['With_Video'] = train_df_copy['微博视频'].apply(lambda x: len(eval(x)))
test_copy['With_Video'] = test_copy['微博视频'].apply(lambda x: len(eval(x)))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
sns.countplot(train_df_copy['With_Video'], ax=ax1, color='grey')
sns.countplot(test_copy['With_Video'], ax=ax2, color='orange')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ax1.set_title('训练集视频分布')
ax2.set_title('测试集视频分布')
plt.show()

print(train_df.情感倾向.value_counts())
train_df = train_df[(train_df.情感倾向 == '1') | (train_df.情感倾向 == '0') | (train_df.情感倾向 == '-1')]
train_df['情感倾向'] = train_df.情感倾向.map({'-1':'0', '0':'1', '1':'2'})

train_df['微博中文内容'].fillna('缺失', inplace=True)
test['微博中文内容'].fillna('缺失', inplace=True)
train_df, test_df = train_test_split(train_df, test_size=0.15, shuffle=True, random_state=4)
train_df, dev_df = train_test_split(train_df, test_size=0.15, shuffle=True, random_state=4)
train_df.to_csv('./train.csv', index=False)
dev_df.to_csv('./dev.csv', index=False)
test_df.to_csv('./test.csv', index=False)
test.to_csv('./test_online.csv', index=False)


#数据清洗
def tokenizer(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*:", "@", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)  # 去除表情符号
    text = re.sub(r"#\S+#","",text)    #  保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)  # 去除网址
    #去掉不在(所有中文、大小写字母、数字)中的非法字符
    regex = re.compile(r'[^\u4e00-\u9fa5A-Za-z0-9]')
    text = regex.sub(' ', text)
    text = text.replace("转发微博","")  # 去除无意义的词语
    text = text.replace("0网页链接?","")
    text = text.replace("?展开全文c", "")
    text = text.replace("网页链接", "")
    text = text.replace("展开全文","")
    text = re.sub(r"\s+"," ", text)# 合并正文中过多的空格
    word_list = [word for word in jieba.cut(text) if word.strip()]
    if len(word_list) == 0:
        word_list.append('缺失')
    #使用jieba分词
    return word_list

TEXT=data.Field(lower=True, tokenize=tokenizer, include_lengths=True)
LABEL=data.Field(sequential=False, use_vocab=False, dtype=torch.long,is_target=True)
ID=data.Field(sequential=False, use_vocab=False)
#创建表格数据集
train, val, test = data.TabularDataset.splits(
    path = './', format = 'csv', skip_header = True,
    train='train.csv', validation='dev.csv', test='test.csv',
    fields=[('微博id',None),
             ('微博发布时间',None),
             ('发布人账号',None),
             ('微博中文内容', TEXT),
             ('微博图片',None),
             ('微博视频',None),
             ('情感倾向', LABEL)]
)
# 线上测试集
test_online = data.TabularDataset(
    path='./test_online.csv', format = 'csv', skip_header = True,
    fields=[('微博id',ID),
               ('微博发布时间',None),
               ('发布人账号',None),
               ('微博中文内容', TEXT),
               ('微博图片',None),
               ('微博视频',None)]
)

TEXT.build_vocab(train, max_size=100000)
LABEL.build_vocab(train)
train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test),
        sort_key=lambda x: len(x.微博中文内容),
        sort_within_batch=True,
        #batch_sizes=(32,32,32),
        batch_sizes=(16,16,16),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
test_online_iter = data.Iterator(
        test_online,
        #batch_size=32,
        batch_size=16,
        sort_key=lambda x: len(x.微博中文内容),
        sort_within_batch=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

for i, batch in enumerate(test_online_iter):
    if i ==1:
        print(batch.微博id.device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

#训练函数
def train_func(model, iterator):
    train_loss = 0
    train_acc = 0
    model.train()
    output_list = []
    cls_list = []
    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
        text, cls = batch.微博中文内容[0], batch.情感倾向
        lengths = batch.微博中文内容[1]
        output = model(text, lengths)  # [B, OUTPUT_SIZE]
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()  # update
        train_loss += loss.item()
        _, output = torch.max(output, 1)
        output_list += output.cpu().numpy().tolist()
        cls_list += cls.cpu().numpy().tolist()
    scheduler.step()
    return train_loss / len(iterator), output_list, cls_list

#评估函数
def evaluate(model, iterator):
    val_loss = 0
    model.eval()
    output_list = []
    cls_list = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            text, cls = batch.微博中文内容[0], batch.情感倾向
            lengths = batch.微博中文内容[1]
            output = model(text, lengths)
            loss = criterion(output, cls)
            val_loss += loss.item()
            _, output = torch.max(output, 1)
            output_list += output.cpu().numpy().tolist()
            cls_list += cls.cpu().numpy().tolist()
    return val_loss / len(iterator), output_list, cls_list


def evaluation_indicators(loss, pre_list, cls_list):
    print('\tLoss: {:.4f}'.format(loss))
    print('\tprecison_score：', precision_score(pre_list, cls_list, average=None, labels=[0, 1, 2]))
    print('\trecall_score：', recall_score(pre_list, cls_list, average=None, labels=[0, 1, 2]))
    print('\tf1_score：', f1_score(pre_list, cls_list, average=None, labels=[0, 1, 2]))
    print('\tf1_score：', f1_score(pre_list, cls_list, average='macro', labels=[0, 1, 2]))
    sns.heatmap(confusion_matrix(pre_list, cls_list), annot=True)
    #plt.show()
    return f1_score(pre_list, cls_list, average='macro', labels=[0, 1, 2])

class Word_AVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_size, output_size)
        self.drop = nn.Dropout(0.4)
    def forward(self, text, lengths):
        embedded = self.embedding(text) # [L, B, EMB_SIZE]
        embedded = embedded.permute(1, 0, 2) # [B, L, EMB_SIZE]
        pooled = torch.mean(embedded, dim=1) # [B, EMB_SIZE]
        pooled = self.drop(pooled)
        out = self.fc(pooled) # [B, OUTPUT_SIZE]
        return out
device = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_SIZE = 512
OUTPUT_SIZE = 3
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
model_word_avg = Word_AVGModel(INPUT_DIM, EMBEDDING_SIZE, OUTPUT_SIZE, PAD_IDX).to(device)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model_word_avg.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)
model_word_avg.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)

N_EPOCHS = 8
best_valid_f1_score = float('inf')
# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss().to(device)
#optimizer = torch.optim.SGD(model_word_avg.parameters(), lr=4.0)
optimizer = torch.optim.SGD(model_word_avg.parameters(), lr=6.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
for epoch in tqdm(range(N_EPOCHS)):
    start_time = time.time()
    train_loss, train_output, train_cls = train_func(model_word_avg, train_iter)
    valid_loss, val_output, val_cls = evaluate(model_word_avg, val_iter)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print('\t 训练集')
    _ = evaluation_indicators(train_loss, train_output, train_cls)
    print('\t 验证集')
    valid_f1_score = evaluation_indicators(valid_loss, val_output, val_cls)
    print('-----------------------------------------------------------------------------------------------------')
    if valid_f1_score < best_valid_f1_score:
        best_valid_f1_score = valid_f1_score
        print('best_valid_f1_score：', best_valid_f1_score)
        print('保存模型')
        torch.save(model_word_avg.state_dict(), 'model_word_avg.pt')
test_loss, test_output, test_cls = evaluate(model_word_avg, test_iter)
print('\t 线下测试集')
evaluation_indicators(test_loss, test_output, test_cls)

def get_result(model, iterator):
    model.eval()
    result = torch.Tensor()
    ID = torch.LongTensor()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            text = batch.微博中文内容[0]
            lengths = batch.微博中文内容[1]
            batch_id = batch.微博id
            ID = torch.cat((ID, batch_id), dim=0)
            output = model(text, lengths)
            result = torch.cat((result, output), dim=0)
    result = result.cpu().numpy()
    ID = ID.cpu().numpy()
    result = result.argmax(axis=1)
    result = pd.DataFrame({'测试数据id': ID, '情感极性':result})
    result['情感极性'] = result.情感极性.map({0:-1, 1:0, 2:1})
    return result

result = get_result(model_word_avg, test_online_iter)
result.to_csv(r'data/result14.csv', index=False)

