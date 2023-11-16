import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

df = pd.read_csv("Result.csv", encoding='utf-8')
column_name = ['index', 'place', 'information', 'category']

data = df.values.tolist()

df = pd.DataFrame(data, columns=column_name)
df

print('전체 문서의 수 :',len(df))

print('NULL 값 존재 유무 :', df.isnull().values.any())

df = df.dropna(how = 'any') # Null 값이 존재하는 행 제거
print('NULL 값 존재 유무 :', df.isnull().values.any()) # Null 값이 존재하는지 확인

df['information'] = df['information'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
df['information']

okt = Okt()

tokenized_data = []
for sentence in tqdm(df['information']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)

print(tokenized_data[:3])

# 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

