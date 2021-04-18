"""
Make inference using bert and lstm crf model

model saving
https://eliyar.biz/nlp_chinese_text_classification_in_15mins/

"""

from os import truncate
from os.path import join as PJ

from kashgari import utils

######### toy model #############
# data
train_x = [["Hello", "world"], ["Hello", "Kashgari"], ["I", "love", "Beijing"]]
train_y = [["O", "O"], ["O", "B-PER"], ["O", "O", "B-LOC"]]

valid_x, valid_y = train_x, train_y
test_x, test_x = train_x, train_y
# model path
SAVE_MODEL_PREFIX = PJ("trained_model", "ner_04-18-17-43-23")


trained_model = utils.load_model(SAVE_MODEL_PREFIX)
#
print(trained_model.predict([train_x[2]], truncating=True))
