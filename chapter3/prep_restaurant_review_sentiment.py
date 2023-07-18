import collections
import numpy as np
import pandas as pd
import re

from argparse import Namespace

args = Namespace(
    raw_train_dataset_csv="/Users/minsoo/Desktop/CS_practice/PyTorch_practice_for_NLP/chapter3/data/raw_train.csv",
    raw_test_dataset_csv="/Users/minsoo/Desktop/CS_practice/PyTorch_practice_for_NLP/chapter3/data/raw_test.csv",
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="/Users/minsoo/Desktop/CS_practice/PyTorch_practice_for_NLP/chapter3/data/reviews_with_splits_lite.csv",
    seed=1337
)

# 원본 데이터를 읽습니다
train_reviews = pd.read_csv(args.raw_train_dataset_csv, header=None, names=['rating', 'review'])
train_reviews = train_reviews[~pd.isnull(train_reviews.review)]
# test_reviews = pd.read_csv(args.raw_test_dataset_csv, header=None, names=['rating', 'review'])
# test_reviews = test_reviews[~pd.isnull(test_reviews.review)]

# 리뷰 클래스 비율이 동일하도록 만듭니다
by_rating = collections.defaultdict(list)
for i, row in train_reviews.iterrows():
    by_rating[row.rating].append(row.to_dict())
    
review_subset = []

for i, item_list in sorted(by_rating.items()):

    n_total = len(item_list)
    n_subset = int(args.train_proportion * n_total)
    review_subset.extend(item_list[:n_subset])

review_subset = pd.DataFrame(review_subset)

# 분할 데이터를 만듭니다.
final_list = []
np.random.seed(args.seed)

for i, item_list in sorted(by_rating.items()):
    np.random.shuffle(item_list)
    n_total = len(item_list)
    n_train = int(n_total * args.train_proportion)
    n_val = int(n_total * args.val_proportion)
    n_test = int(n_total * args.test_proportion)

    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train + n_val]:
        item['split'] = 'valid'
    for item in item_list[n_train + n_val: n_train + n_val + n_test]:
        item['split'] = 'test'

    # 최종 리스트에 추가합니다
    final_list.extend(item_list)

final_reviews = pd.DataFrame(final_list)

def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r'([.,!?])', r' \1 ', text)
    text = re.sub(r'[^a-zA-Z.,!?]+', r" ", text)
    return text

final_reviews.review = final_reviews.review.apply(simple_preprocess)
final_reviews.to_csv(args.output_munged_csv, index=False)
