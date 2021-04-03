"""
this function prodivde in kashgari NER colabl tutorial
https://colab.research.google.com/drive/1yKo5h1Eszou5_W18-BQvgqGuzK6uyEnd#scrollTo=4LqUOxB0LbmE

all bert pre-trained model : 
https://github.com/google-research/bert

about zipfile in python
https://clay-atlas.com/blog/2020/08/13/python-%E4%BD%BF%E7%94%A8-zipfile-%E5%A3%93%E7%B8%AE%E3%80%81%E8%A7%A3%E5%A3%93%E8%B3%87%E6%96%99%E5%A4%BE/

model performance benchmark
https://kashgari.readthedocs.io/en/v2.0.1/tutorial/text-labeling/#chinese-ner-performance

"""
import os
import zipfile
from os.path import abspath, dirname, join

from tensorflow.keras.utils import get_file

HERE = abspath(dirname(__file__))
EMBEDDING_FOLDER = join(HERE, "language_model")
BERT_PATH = ""


def download_bert_if_needs(parent_dir: str) -> str:
    bert_path = os.path.join(parent_dir, "chinese_L-12_H-768_A-12")
    # BERT-Base, Chinese: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
    if not os.path.exists(bert_path):
        zip_file_path = get_file(
            "chinese_L-12_H-768_A-12.zip",
            "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip",
            untar=False,
            cache_subdir="",
            cache_dir=parent_dir,
        )
        unzipped_file = zipfile.ZipFile(zip_file_path, "r")
        unzipped_file.extractall(path=parent_dir)
    return bert_path


if BERT_PATH == "":
    BERT_PATH = download_bert_if_needs(os.path.join(EMBEDDING_FOLDER, "bert"))
print("BERT    : ", BERT_PATH)

