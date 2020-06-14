from PIL import Image
import os, glob
import numpy as np
import random, math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

class DataManager:
    #画像が保存されているルートディレクトリのパス
    root_dir = "data"
    # 商品名
    categories = ["tsutenkaku","taiyo_no_to"]

    # 画像データ用配列
    X = []
    # ラベルデータ用配列
    Y = []

    #全画像データ(categoryに対応する番号,画像名?のセット)格納用配列
    allfiles = []

    #画像データごとにadd_sample()を呼び出し、X(画像np配列データ),Y(カテゴリーインデックス)の配列を返す関数
    def make_sample(self,files):
        global X, Y
        X = []
        Y = []
        for cat, fname in files:
            add_sample(cat, fname)
        return np.array(X), np.array(Y)

    #渡された画像データを読み込んでXに格納し、また、
    #画像データに対応するcategoriesのidxをYに格納する関数
    def add_sample(self,cat, fname):
        img = Image.open(fname)
        img = img.convert("RGB")
        img = img.resize((250, 250))
        data = np.asarray(img)
        X.append(data) #global変数x
        Y.append(cat)

    def getFiles(self):
        #カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
        for idx, cat in enumerate(categories):
            image_dir = root_dir + "/" + cat #例えばdata/tsutenkaku
            files = glob.glob(image_dir + "/*.jpg") #ファイルの取得
            for f in files:
                allfiles.append((idx, f))

    def arrangeData(self):
        getFiles()

        """"""
        #シャッフル後、学習データ80%と検証データ20%に分ける
        random.shuffle(allfiles)
        th = math.floor(len(allfiles) * 0.8)
        train = allfiles[0:th] #学習データ
        test  = allfiles[th:] #学習データ
        X_train, y_train = make_sample(train) # X(画像np配列データ),Y(カテゴリーインデックス)
        X_test, y_test = make_sample(test)
        xy = (X_train, X_test, y_train, y_test)
        #データを保存する（データの名前を「img_data.npy」としている）
        np.save("img_data.npy", xy)
        """"""
class Learning:
    def createModel(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(250,250,3)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64,(3,3),activation="relu"))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(128,(3,3),activation="relu"))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(128,(3,3),activation="relu"))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(512,activation="relu"))
        model.add(layers.Dense(3,activation="sigmoid")) #分類先の種類分設定

        #モデル構成の確認
        model.summary()
class TfDebug:
    def testGPURecognization(self):
        #TensorFlowがGPUを認識しているか確認
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
def main():
    #GPUが認識されているか確認
    tfdebug=TfDebug()
    tfdebug.testGPURecognization()


if __name__ == "__main__":
    main()


