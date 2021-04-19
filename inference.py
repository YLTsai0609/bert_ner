"""
Make inference using bert and lstm crf model

model saving
https://eliyar.biz/nlp_chinese_text_classification_in_15mins/

GPU ~ 100ms
CPU ~ 150ms
"""

import os
import random
import time
from os.path import join as PJ

import numpy as np
from kashgari import utils
from kashgari.corpus import ChineseDailyNerCorpus

SEED = 42
WARM_START = 10
N_INFERENCE = 50
random.seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # if you wabt to run usingh CPU
######### toy model #############
# # data
# train_x = [["Hello", "world"], ["Hello", "Kashgari"], ["I", "love", "Beijing"]]
# train_y = [["O", "O"], ["O", "B-PER"], ["O", "O", "B-LOC"]]

# valid_x, valid_y = train_x, train_y
# test_x, test_x = train_x, train_y
# # model path
# SAVE_MODEL_PREFIX = PJ("trained_model", "ner_04-18-17-43-23")
######### Chinese News ###########
test_x, test_y = ChineseDailyNerCorpus.load_data("test")
SAVE_MODEL_PREFIX = PJ("trained_model", "ner_daily_news")

############ start profiling ##############
trained_model = utils.load_model(SAVE_MODEL_PREFIX)
for _ in range(WARM_START):
    trained_model.predict([test_x[2]], truncating=True)

inf_time = []
for i in range(N_INFERENCE):
    start = time.time()
    trained_model.predict([test_x[2]], truncating=True)
    end = time.time()
    inf_ms = (end - start) * 1000
    inf_time.append(inf_ms)
mean, std = np.array(inf_time).mean(), np.array(inf_time).std()
############ prediction samples ###########

smaples = random.choices(
    test_x,
    k=10,
)
for sentence in smaples:
    print(sentence)
    print(trained_model.predict([sentence], truncating=True))
print("Inference stats (mean, std): ", mean, std)
