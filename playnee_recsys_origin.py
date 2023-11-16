# -*- coding: utf-8 -*-
"""playnee_recsys.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LECHlSrz3saludswVa4X_Wty-nijkDEl
"""

#pip install gensim==4.3.0

import gensim
gensim.__version__

#pip install konlpy
#pip install mecab-python
#코랩환경 기준
#from google.colab import drive
#drive.mount('/content/my_home')

#from konlpy.tag import Mecab
#mecab = Mecab()

"""# 1. 데이터 로드"""

import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

#nltk.download('stopwords')
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

from gensim.models import Word2Vec
model = Word2Vec(tokenized_data, window=5, min_count=5, workers=4, sg=0)

print('완성된 임베딩 매트릭스의 크기 확인 :', model.wv.vectors.shape)


df[:5]

import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from PIL import Image
from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

"""# 2. 사전 훈련된 워드 임베딩 사용하기"""


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

"""# 3. 단어 벡터의 평균 구하기"""

def get_document_vectors(document_list):
    document_embedding_list = []

    # 각 문서에 대해서
    for line in document_list:
        doc2vec = None
        count = 0
        for word in line.split():
            if word in word2vec_model.wv.vocab:
                count += 1
                # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is None:
                    doc2vec = word2vec_model[word]
                else:
                    doc2vec = doc2vec + word2vec_model[word]

        if doc2vec is not None:
            # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠준다.
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)
    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return document_embedding_list
   

document_embedding_list = get_document_vectors(df['information'])
print('문서 벡터의 수 :',len(document_embedding_list))

"""# 4. 추천 시스템 구현하기"""

cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)

cosine_similarities[0]

print('코사인 유사도 매트릭스의 크기 :',cosine_similarities.shape)

def recommendations(places):
    # Initialize an empty list to store recommendations
    all_recommendations = []
    all_score = []
    for place in places:
        travel = df[['place', 'category', 'information']]

        # When a location is entered, the index of the location is returned and stored in idx.
        indices = pd.Series(df.index, index=df['place']).drop_duplicates()
        idx = indices[place]

        # Select a similar document embedding.
        sim_scores = list(enumerate(cosine_similarities[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[0:10]  # Take top 10 recommendations
        print(sim_scores)
        # Most similar index
        all_score.extend(sim_scores)

    print(all_score)    
    travel_indices = [i[0] for i in all_score]
    print(travel_indices)
        
        # Extract only the rows of the corresponding index from the entire data frame. It has 5 rows.
    recommend = travel.iloc[travel_indices].reset_index(drop=True)
        # Create a list of recommended locations for the current place
    locations = recommend['place'].tolist()

        # Add the recommendations for the current place to the overall list
    all_recommendations.extend(locations)
    top_5_items = all_recommendations[:5]

    print(top_5_items)
    # Remove duplicate locations
    # Sort the recommendations based on cosine similarity in descending order
    #sorted_recommendations = sorted(unique_recommendations, key=lambda x: cosine_similarity(x, places), reverse=True)
#    sorted_recommendations = sorted(unique_recommendations, key=lambda x: cosine_similarities[unique_recommendations.index(x)], reverse=True)

    # Generate response in JSON format

# Example usage
da =['반포대교 야경','경복궁','매봉산 야경']
recommendations(da)



