import torch
import numpy as np
import spacy
import nltk

# 자연어 처리 (Natural Language Processing)?
# 전산 언어학과 마찬가지로 컴퓨터를 사용한 언어 연구 분야
# 전산 언어학 (Computational Linguistics) -> 언어의 특징을 이해하는 방법을 개발 Ex) 음운론, 형태론, 구분론, 의미론, 화용론

# 2.1 말뭉치, 토큰, 타입
# 모든 NLP 작업은 말뭉치(Corpus, Corpora - 복수)라 부르는 Text Data로 시작
# Corpus는 일반적으로 원시 텍스트(ASCII, UTF-8)와 이 텍스트에 연관된 Meta Data(Data에 대한 정보를 담은 Data)를 포함
# 원시 텍스트는 문자 (Byte) 시퀀스지만 일반적으로 문자를 토큰(Token)이라는 연속된 단위로 묶었을 때 사용
# 샘플(Sample), 데이터 포인트(Data Point) ?
# ML 분야에서는 Meta Data가 붙은 모든 Text를 샘플 또는 데이터 포인트라고 지칭함
# 샘플의 모음인 말뭉치(Corpus)는 데이터셋(Dataset)이라고 지칭함

# 토큰화(Tokenization) - Text를 Token으로 나누는 과정
# 토큰화의 기준은 저마다 다름

nlp = spacy.load('en_core_web_sm') # BTW, spaCy 오류로 tensorflow의 __init__.py 481 번째 줄 _keras._load() -> keras._load()로 바꿈.. 참고
text = "Mary, don't slap the green witch"
print([str(token) for token in nlp(text.lower())])

tweet = u"Snow White and the Seven Degrees #MakeAMovieCold@midnight:-)"
tokenizer = nltk.tokenize.TweetTokenizer()
print(tokenizer.tokenize(tweet))

# 타입(Type)은 말뭉치에 등장하는 고유한 Token
# 말뭉치(Courpus)에 있는 모든 타입의 집합이 어휘 사전 또는 어휘(Lexicon)
# 단어(Word)의 구성 - 내용어(Content Words)
#                - 불용어(Stopword) -> 관사와 전치사등 내용어를 보충하는 문법적인 용도

# 특성 공학(Feature Engineering) - 언어학을 이해하고 NLP 문제 해결에 적용하는 과정

# 2.2 n-Gram: 유니그램, 바이그램, 트라이그램...
# -> Text에 있는 고정 길이(n)의 연속된 토큰 시퀀스 - Unigram : token 1개
#                                         - Bigram  : token 2개
#                                         - Trigram : token 3개
# 부분단어(Subword) 자체가 유용하다면, 문자 n-gram을 생성할 수 있음
# Ex) methanol의 접미사 -ol은 알코올의 종류를 나타냄 -> 유기화합물 구분 작업에서는 n-gram으로 찾은 부분단어의 정보 유용
# -> 모든 문자의 n-gram을 토큰 하나로 취급시 용이
def n_grams(text, n):
    result = []
    for i in range(len(text)-n+1):
        result.append(text[i:i+n])
    return result

cleaned = ['mary', ',', 'n\'t', 'slap', 'green', 'witch', '.']
print(n_grams(cleaned, 3))

# 2.3 표제어(Lemma)와 어간(Stem)
# 표제어는 단어의 기본형 Ex) fly는 flow, flew, flies, flown, flowing들의 표제어
# 토큰을 표제어로 바꾸어 Vector 표현의 Dimension을 줄이는 방법이 종종 도움이 됨 -> 이런 축소를 표제어 추출(Lemmatization)이라고 함
# 단어의 끝을 잘라 어간이라는 공통 형태로 축소 -> 어간 추출(Stemming)
doc = nlp(u"he was running late")
for token in doc:
    print(f'{token} -> {token.lemma_}')
print('--------------------------------')

# 2.4 문장과 문서 분류하기
# 문서(Document) 분서 작업은 NLP 분야 초기 application 중 하나 -> TF/TF-IDF 표현이 문서나 문장 같은 긴 텍스트 뭉치 분류 도움
# Topic 레이블 할당, 리뷰의 감성 예측, 스팸 이메일 필터링, 언어 식별, 이메일 분류 같은 작업은 지도 학습(Supervised Learning)기반의 문서 분류 문제

# 2.5 단어 분류하기: 품사 태깅
# 단어 분류 작업의 예시 - 품사(Part of Speech - POS) 태깅(Tagging)
doc = nlp(u"Mary slaped the green witch.")
for token in doc:
    print(f'{token} -> {token.pos_}')
print('--------------------------------')

# 2.6 청크 나누기와 개체명 인식
# 종종 연속된 여러 토큰으로 구분되는 텍스트 구(phrase)에 레이블(label)을 할당해야함 Ex) 명사구(NP)와 동사구(VP) 구별
# 이를 청크 나누기 (Chunking) 또는 부분 구문 분석(Shallow Parsing)이라고 함
doc = nlp(u"Mary slaped the green witch.")
for token in doc.noun_chunks:
    print(f'{token} -> {token.label_}')
print('--------------------------------')

# 2.7 문장 구조
# 구 단위를 식별하는 부분 구문 분석(Shallow Parsing)과 달리 '구 사이의 관계'를 파악하는 작업을 구문 분석(Parsing)이라고 함
# 구문 분석 트리 (Parse Tree)는 문장 안의 문법 요소가 계층적으로 어떻게 관련되는지 보여줌 - 구성 구문 분석(Constituent Parsing)
#                                                                       - 의준 구문 분석(Dependancy Parsing)

# 2.8 단어 의미와 의미론
# 단어에는 의미가 하나 이상 있음. 단어가 나태내는 각각의 뜻을 단어의 의미(Sense) 라고 지칭함
# 단어 의미는 문맥으로 결정될 수도 있음. -> NLP에 적용된 첫 번째 준지도 학습(Semi-Supervised Learning) -> 단어 의미를 자동으로 찾는 일 이었음

