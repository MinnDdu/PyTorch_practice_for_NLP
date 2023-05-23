from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

corpus = ['Time flies like an arrow', 
          'Fruit flies like a banana']

# 1. One-Hot vecotorinzing % TF (Term Frequency)
# {time, fruit, flies, like, a, an, arrow, banana}
# [1,   1,  2,  2,  1,  1,  1,  1] -> TF representation of corpus above / It is also refers to the BoG (Bag of Words)
# TF -> 단어(word)의 등장 횟수 (Frequency)에 비례하여 단어에 가중치를 부여 
one_hot_vectorizer = CountVectorizer(binary=True) # for one-got encoding, we use 'binary=True'
one_hot_encoding = one_hot_vectorizer.fit_transform(corpus).toarray()
vocabulary = one_hot_vectorizer.get_feature_names_out()
sns.heatmap(one_hot_encoding, annot=True, cbar=False, xticklabels=vocabulary, yticklabels=['Sentence 1', 'Sentence 2'])
plt.show()

# TF-IDF (Term Frequency - Inverse Document Frequency)
# 가장 흔한 단어 가중치 최하 / 가장 희소성 높은 단어 가중치 최대 ->> IDF (Inverse Document Frequency) 적합
# IDF는 흔한 토큰(단어)의 점수 낮추고 드문 토큰의 점수를 높임
# IDF(w) = log(N/n_w) / N = 전체 documents 개수, n_w = 단어 w를 포함한 documents 개수
# TF-IDF = TF(w) * IDF(w) -> 매우 흔한 단어 (N == n_w)는 IDF(w) == 0 따라서 TF(w) * IDF(w) == 0
# ** Scikit Learn 의 IDF(w) = log((N+1) / (n_w+1)) + 1 계산 후, IDF(w) 값 L2 Norm으로 정규화 한 값 산출
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
sns.heatmap(tfidf, annot=True, cbar=False, xticklabels=vocabulary, yticklabels=['Sentence 1', 'Sentence 2'])
plt.show()

