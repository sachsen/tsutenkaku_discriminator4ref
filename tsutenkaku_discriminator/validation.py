
import numpy as np
from PIL import Image
import os, glob

import random, math
import learning
from tensorflow.compat.v1.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import pickle

def dataManage():


    # 画像が保存されているディレクトリのパス
    root_dir = "./data/"

    dm= learning.DataManager()
    categories = dm.getCategory()

    X = [] # 画像データ
    Y = [] # ラベルデータ

    # フォルダごとに分けられたファイルを収集
    #（categoriesのidxと、画像のファイルパスが紐づいたリストを生成）
    allfiles = []
    for idx, cat in enumerate(categories):
        image_dir = root_dir + "/" + cat#+"_valid"
        files = glob.glob(image_dir + "/*.jpg")
        for f in files:
            allfiles.append((idx, f))

    for cat, fname in allfiles:
        img = Image.open(fname)
        img = img.convert("RGB")
        img = img.resize((250, 250))
        data = np.asarray(img)
        X.append(data)
        Y.append(cat)

    x = np.array(X)
    y = np.array(Y)

    np.save("./data/param/validation_data_test_X_250.npy", x)
    np.save("./data/param/validation_data_test_Y_250.npy", y)
    return categories
def evaluate(category):
    # モデルの精度を測る

    #評価用のデータの読み込み
    eval_X = np.load("./data/param/validation_data_test_X_250.npy").astype("float")/255
    eval_Y = np.load("./data/param/validation_data_test_Y_250.npy").astype("float")/255

    #Yのデータをone-hotに変換
    

    test_Y =to_categorical(eval_Y, len(category))
    json_string = open('./data/param/optimizedmodel.json').read()

    model = model_from_json(json_string)
    with open("./data/param/bestparam.pickle", mode='rb') as f:
            bestparam = pickle.load(f)
    model.compile(optimizer=bestparam["optimizer"],
                loss="binary_crossentropy",
                metrics=["accuracy"])
    hdf5_file = "./data/param/optimizedModelWeight.hdf5"
    model.load_weights(hdf5_file)
    print(model.summary())
    score = model.evaluate(x=eval_X,y=test_Y)

    print('loss=', score[0])
    print('accuracy=', score[1])

def main():
    category= dataManage()
    evaluate(category)
    pass
if __name__ == "__main__":
    main()