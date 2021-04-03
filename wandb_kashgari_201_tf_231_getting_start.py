"""
https://kashgari.readthedocs.io/en/v2.0.1/tutorial/text-labeling/

https://github.com/BrikerMan/Kashgari

https://docs.wandb.ai/ref/keras


"""
import os
from os.path import join as PJ

import kashgari
import tensorflow as tf
import wandb
from kashgari.callbacks import EvalCallBack
from kashgari.corpus import ChineseDailyNerCorpus
from kashgari.embeddings.bert_embedding import BertEmbedding
from kashgari.tasks.labeling import (
    BiLSTM_CRF_Model,
)  # 还可以选择 `CNN_LSTM_Model`, `BiLSTM_Model`, `BiGRU_Model` 或 `BiGRU_CRF_Model`
from wandb.keras import WandbCallback

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
        "batch_size": 512,
        "epochs": 20,
    },
)
config = wandb.config


PRE_TRAINED = PJ("language_model", "bert", "chinese_L-12_H-768_A-12")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # if you wabt to run usingh CPU
print(
    "tesnrflow version : ", tf.__version__, "kashgari version : ", kashgari.__version__
)

train_x, train_y = ChineseDailyNerCorpus.load_data("train")
valid_x, valid_y = ChineseDailyNerCorpus.load_data("validate")
test_x, test_y = ChineseDailyNerCorpus.load_data("test")

print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(valid_x)}")
print(f"test data count: {len(test_x)}")


bert_embed = BertEmbedding(PRE_TRAINED, sequence_length=config.sequence_length)

model = BiLSTM_CRF_Model(bert_embed)

callbacks = [
    EvalCallBack(kash_model=model, valid_x=valid_x, valid_y=valid_y, step=1),
    WandbCallback(log_batch_frequency=1),
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
