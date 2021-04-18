"""
tutorial

https://kashgari.readthedocs.io/en/v2.0.1/tutorial/text-labeling/

src
https://github.com/BrikerMan/Kashgari

callback
https://docs.wandb.ai/ref/keras

model saving
https://eliyar.biz/nlp_chinese_text_classification_in_15mins/

Command : /home/joetsai/.conda/envs/py37_tf231/bin/python wandb_kashgari_201_tf_231.py


"""
import datetime
import json
import os
from os.path import join as PJ

import kashgari
import tensorflow as tf
from kashgari import utils
from kashgari.callbacks import EvalCallBack
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings.bert_embedding import BertEmbedding
from kashgari.tasks.labeling import (
    BiLSTM_CRF_Model,
)  # 还可以选择 `CNN_LSTM_Model`, `BiLSTM_Model`, `BiGRU_Model` 或 `BiGRU_CRF_Model`
from tensorflow import keras

import wandb
from f1_wandbcallback import F1_WandbCallback


############## utils function ####################
def now():
    return datetime.datetime.now().strftime("%m-%d-%H-%M-%S")


############## limit gpu resource #############
# gpus = tf.config.experimental.list_physical_devices("GPU")
# print(gpus)
# usage_gb = 2.5
# tf.config.experimental.set_virtual_device_configuration(
#    gpus[0],
#    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=usage_gb * 1024)],
# )

############ wandb setting ################

os.environ["WANDB_MODE"] = "dryrun"

wandb.init(
    project="bert-ner",
    config={
        "dataset": "ChineseDailyNerCorpus",
        "sequence_length": 100,
        "batch_size": 128,
        "epochs": 15,
        "early_stop": 10,
        "reduce_lr_plateau_steps": 5,
        "reduce_lr_factor": 0.1,
    },
)
config = wandb.config


PRE_TRAINED = PJ("language_model", "bert", "chinese_L-12_H-768_A-12")
SAVE_MODEL_PREFIX = "trained_model"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # if you wabt to run usingh CPU
print(
    "tesnrflow version : ", tf.__version__, "kashgari version : ", kashgari.__version__
)

# ChineseDailyNerCorpus
# train_x, train_y = ChineseDailyNerCorpus.load_data("train")
# valid_x, valid_y = ChineseDailyNerCorpus.load_data("validate")
# test_x, test_y = ChineseDailyNerCorpus.load_data("test")

# Or use your own corpus
train_x = [["Hello", "world"], ["Hello", "Kashgari"], ["I", "love", "Beijing"]]
train_y = [["O", "O"], ["O", "B-PER"], ["O", "O", "B-LOC"]]

valid_x, valid_y = train_x, train_y
test_x, test_x = train_x, train_y

print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(valid_x)}")
print(f"test data count: {len(test_x)}")


bert_embed = BertEmbedding(
    PRE_TRAINED, sequence_length=config.sequence_length, batch_size=config.batch_size
)


model = BiLSTM_CRF_Model(
    bert_embed,
    sequence_length=config.sequence_length,
)

eval_callback = EvalCallBack(
    kash_model=model, x_data=valid_x, y_data=valid_y, step=1, truncating=True
)
f1_wandb_callback = F1_WandbCallback(
    kashgari_eval_callback=eval_callback,
    log_batch_frequency=1,
    save_model=False,
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=config.early_stop),
    keras.callbacks.ReduceLROnPlateau(
        factor=config.reduce_lr_factor, patience=config.reduce_lr_plateau_steps
    ),
    f1_wandb_callback,
]

model.fit(
    train_x,
    train_y,
    x_validate=valid_x,
    y_validate=valid_y,
    epochs=config.epochs,
    batch_size=config.batch_size,
    callbacks=callbacks,
)


result_alias = f"ner_{now()}"
with open(PJ("expr_result", f"{result_alias}.json"), "w") as f:
    f.write(json.dumps(eval_callback.logs, indent=2))

# will save bert h5, downstream model h5, and model_config.json in folder
model.save(PJ(SAVE_MODEL_PREFIX, f"{result_alias}"))
