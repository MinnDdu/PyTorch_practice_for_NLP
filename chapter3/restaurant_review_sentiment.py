from argparse import Namespace
from collections import Counter
import json
import os
import re
import string
import collections

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm

# 긍정/부정으로 이루어진 감성 레이블과 리뷰가 쌍을 이룬 데이터셋인 Yelp 데이터셋 이용, 긍정 부정 리뷰 분류
