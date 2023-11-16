
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

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



"""# 2. 사전 훈련된 워드 임베딩 사용하기"""


word2vec_model = Word2Vec(size = 300, window=5, min_count = 2, workers = -1)
word2vec_model.build_vocab(tokenized_data)
#word2vec_model.intersect_word2vec_format('/content/my_home/MyDrive/Colab Notebooks/ko.bin', lockf=1.0, binary=True)
word2vec_model.train(tokenized_data, total_examples = word2vec_model.corpus_count, epochs = 15)

model_filename = "word2vec_model.bin"
word2vec_model.save(model_filename)

"""


    word2vec_model = Word2Vec(size = 300, window=5, min_count = 2, workers = -1)
    word2vec_model.build_vocab(tokenized_data)
    #word2vec_model.intersect_word2vec_format('/content/my_home/MyDrive/Colab Notebooks/ko.bin', lockf=1.0, binary=True)
    word2vec_model.train(tokenized_data, total_examples = word2vec_model.corpus_count, epochs = 15)

    #from gensim.models import Word2Vec
    #from gensim.models import KeyedVectors

    # Word2Vec 모델 초기화
    #word2vec_model = Word2Vec(size=300, window=5, min_count=2, workers=-1)
    #word2vec_model.build_vocab(tokenized_data)

    # 한국어 워드 임베딩 불러오기
    #ko_word_vectors = KeyedVectors.load_word2vec_format('/content/my_home/MyDrive/Colab Notebooks/ko.bin')

    # Word2Vec 모델에 한국어 워드 임베딩 추가
    #word2vec_model.wv.add(ko_word_vectors)

    # 나머지 훈련 코드
    #word2vec_model.train(tokenized_data, total_examples=word2vec_model.corpus_count, epochs=15)

"""