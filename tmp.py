from flask import Flask, request, jsonify

#필요한 모듈 호출
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_request():
            # JSON 데이터를 파싱
            data = request.get_json()

            # JSON 데이터에서 필요한 정보 추출
            user_preferred_places = data.get('userPreferredPlaces')
            plan_preferred_places = data.get('planPreferredPlaces')
            additional_condition = data.get('additionalCondition')
            all_places = data.get('allPlaces')
            #제외할 카테고리 파싱 추가 beta버전
            #excluded_categories = data.get('excludedCategory')
            
            #json 호출 형태 확인
            print(user_preferred_places)
            print(plan_preferred_places)
            print(additional_condition)
            #print(all_places)
            wantplaces = user_preferred_places.extend(plan_preferred_places)
            print(wantplaces)
            excluded_categories = []
            stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
            df = pd.read_csv("Result.csv", encoding='utf-8')
            column_name = ['id', 'place', 'information', 'category']
            # Create DataFrame
            data = df.values.tolist()
            df = pd.DataFrame(data, columns=column_name)
            
            column_name2 = ['name', 'distance']
            df2 = pd.DataFrame(list(all_places.items()), columns=column_name2)
            filtered_ids = df2[df2['distance'] <= 25000]['name'].tolist()            

            print('전체 문서의 수 :',len(df)) #df 확인
            df = df.dropna(how = 'any') # Null 값이 존재하는 행 제거
            print('NULL 값 존재 유무 :', df.isnull().values.any()) # Null 값이 존재하는지 확인

            df['information'] = df['information'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
            
            #사용하는 모델 로드
            loaded_word2vec_model = Word2Vec.load('word2vec_model.bin')
            word2vec_model = loaded_word2vec_model
            
            #장소의 문서정보 벡터화
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
                                    doc2vec = word2vec_model.wv[word]
                                else:
                                    doc2vec = doc2vec + word2vec_model.wv[word]

                        if doc2vec is not None:
                            # 단어 벡터를 모두 더한 벡터의 값을 문서정보 길이로 나눠준다.
                            doc2vec = doc2vec / count
                            document_embedding_list.append(doc2vec)
                    # 각 문서에 대한 문서 벡터 리스트를 리턴
                    return document_embedding_list


            document_embedding_list = get_document_vectors(df['information'])

            """# 4. 추천 시스템 구현하기"""
            #코사인유사도로 장소의 문서정보 계산
            cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)

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
                        sim_scores = sim_scores[0:7]  # Take top 10 recommendations
                        print(sim_scores)
                        # Most similar index
                        all_score.extend(sim_scores)
                    travel_indices = [i[0] for i in all_score]
                    recommend = travel.iloc[travel_indices].reset_index(drop=True)
                    locations = recommend['place'].tolist()
                    all_recommendations.extend(locations)
                    top_5_items = all_recommendations[:5]

                    return top_5_items
            #형태소 한글 토큰화
            okt = Okt()

            def tokenize(sentence):
                tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
                stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
                return stopwords_removed_sentence

            def recommend_similar_places(input_text, df, word2vec_model, cosine_similarities,excluded_categories):
                # 입력된 텍스트를 토큰화
                tokenized_input = tokenize(input_text)
                # 입력된 텍스트의 문서 벡터 계산
                input_vector = sum([word2vec_model.wv[word] for word in tokenized_input if word in word2vec_model.wv.vocab])
                df = df[df['place'].isin(filtered_ids)]
                # 모든 문서와의 코사인 유사도 계산
                similarities = cosine_similarity([input_vector], document_embedding_list)[0]
                
                # 유사도가 높은 상위 5개의 장소 추천
                top_indices = similarities.argsort()[-20:][::-1]
                top_recommendations = df.iloc[top_indices][['place', 'category', 'information']].reset_index(drop=True)
                # 특정 카테고리 제외
                top_recommendations = top_recommendations[~top_recommendations['category'].isin(excluded_categories)]
                #print('추천 값:',top_recommendations)
                top_recommendations = top_recommendations.head(5)
                locations = top_recommendations['place'].tolist()
                #print('장소 값:',locations)
                
                return locations
            
            if additional_condition == None:
                return jsonify(recommendations(wantplaces))

            else:
                return jsonify(recommend_similar_places( additional_condition , df, word2vec_model,cosine_similarities,excluded_categories ))
            
           
if __name__ == '__main__':
    app.run(host = '0.0.0.0' , port = '5000' ,debug=True)
