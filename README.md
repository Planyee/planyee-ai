# planyee-ai
플래니(planyee) AI 서버


# 데이터 전처리
가지고있는 장소 정보 문서
Result.csv
장소 문서 정보 토크나이징
planyee_tokenize.py
토큰의 벡터화 및 문서 벡터화
word2vec_token.py

# word2vec 모델
gensim 라이브러리 활용 version 3.8
ko.bin - 한글 학습 word2vec
word2vec_model.bin - result.csv 학습 word2vec

# 초기 Ai 데이터 문서화 recsys 구현
*** 장소의 문서정보를 이용하여 코사인유사도 값으로 추천 ***
planyee_recsys.ipynb
planyee_recsys.py

# 카테고리 적용
*** 장소의 문서정보를 이용하여 코사인유사도 값으로 추천 ***
*** 장소의 문서정보의 카테고리와 사용자의 text 값으로 추천 ***


# 현재 사용 중인 모듈
tmp.py
