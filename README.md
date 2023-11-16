# planyee-ai
플래니(planyee) AI 서버
### 데이터 전처리
- Result.csv -- 가지고있는 장소 정보 문서
- planyee_tokenize.py -- 장소 문서 정보 토크나이징
- word2vec_token.py -- 토큰의 벡터화 및 문서 벡터화
### word2vec 모델
- gensim 라이브러리 활용 version 3.8
- ko.bin - 한글 학습 word2vec
- word2vec_model.bin - result.csv 학습 word2vec
### 초기 Ai 데이터 문서화 recsys 구현
- **장소의 문서정보를 이용하여 코사인유사도 값으로 추천**
- planyee_recsys.ipynb
- planyee_recsys.py
### 현재 구현완료 된 py 파일
- **장소의 문서정보만을 이용하여 코사인유사도 값으로 추천***
- - planyee_recsys_origin.py
- **장소의 문서정보의 카테고리와 사용자의 freetext(가고싶은 장소에 대한 설명) 값으로 추천***
## 현재 사용중인 모듈
- tmp.py
# 향 후 계획
- 이를 통해 추천시스템의 coldstartness와 sparsity 단점을 매꿔서 처음 가입한 사용자에게 위치정보를 추천해주고, 이에 대한 데이터를 쌓아 개인화 추천시스템 구축
- 개인화 추천시스템은 GCN으로 위치정보와 사용자간의 지식그래프를 활용
