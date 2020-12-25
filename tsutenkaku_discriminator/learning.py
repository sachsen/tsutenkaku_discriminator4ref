from PIL import Image
import os, glob
import numpy as np
import random, math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models,optimizers
from tensorflow.compat.v1.keras.utils import to_categorical
import optuna
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.layers import Convolution2D, Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle

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
    #外部参照用
    def getCategory(self):
        return self.categories
    #画像データごとにadd_sample()を呼び出し、X(画像np配列データ),Y(カテゴリーインデックス)の配列を返す関数
    def make_sample(self,files):
        global X, Y
        X = []
        Y = []
        for cat, fname in files:
            self.add_sample(cat, fname)
        return np.array(X).astype(np.float16)/255, np.array(Y) # ついでにデータの正規化をする。

    #渡された画像データを読み込んでXに格納し、また、
    #画像データに対応するcategoriesのidxをYに格納する関数
    def add_sample(self,cat, fname):
        img = Image.open(fname)
        img = img.convert("RGB")
        #img = img.resize((250, 250))#リサイズは画像が引き延ばされてしまうという欠点がある。
        img=self.expand2square(img, (0, 0, 0)).resize((250, 250), Image.LANCZOS)
        #img.show()

        data = np.asarray(img)
        X.append(data) #global変数x
        Y.append(cat)

    def getFiles(self):
        self.count=0
        #カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
        for idx, cat in enumerate(self.categories):
            image_dir = self.root_dir + "/data_generated_" + cat #例えばdata/data_generated_tsutenkaku
            files = glob.glob(image_dir + "/*.jpeg") #ファイルの取得
            for f in files:
                self.allfiles.append((idx, f))
                self.count+=1
        for idx, cat in enumerate(self.categories):
            image_dir = self.root_dir + "/data_valid1_" + cat #例えばdata/tsutenkaku
            files = glob.glob(image_dir + "/*.jpeg") #ファイルの取得
            for f in files:
                self.allfiles.append((idx, f))

    def arrangeData(self):
        self.getFiles()

        
        #シャッフル後、学習データ80%と検証データ20%に分ける
        random.shuffle(self.allfiles[0:self.count])
        random.shuffle(self.allfiles[self.count:])
        th = math.floor(len(self.allfiles) * 0.8)
        train = self.allfiles[0:self.count] #学習データ
        test  = self.allfiles[self.count:] #テストデータ
        X_train, y_train = self.make_sample(train) # X(画像np配列データ),Y(カテゴリーインデックス)
        X_test, y_test = self.make_sample(test)
        xy = (X_train, X_test, y_train, y_test)
        #データを保存する（データの名前を「img_data.npy」としている）
        with open("./data/temp/img_x_train.pickle", mode='wb') as f:
            pickle.dump(X_train,f,protocol=pickle.HIGHEST_PROTOCOL)
        with open("./data/temp/img_y_train.pickle", mode='wb') as f:
            pickle.dump(y_train,f,protocol=pickle.HIGHEST_PROTOCOL)
        with open("./data/temp/img_x_test.pickle", mode='wb') as f:
            pickle.dump(X_test,f,protocol=pickle.HIGHEST_PROTOCOL)
        with open("./data/temp/img_y_test.pickle", mode='wb') as f:
            pickle.dump(y_test,f,protocol=pickle.HIGHEST_PROTOCOL)


        print("img_data saved!")
        return self.categories
        
    #引用 https://note.nkmk.me/python-pillow-square-circle-thumbnail/
    def expand2square(self,pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
class Learning:
    def __init__(self, category):
        with open("./data/temp/img_x_train.pickle", mode='rb') as f:#withはスコープを生成しない
           self.x_train = pickle.load(f)
        with open("./data/temp/img_y_train.pickle", mode='rb') as f:
           self.y_train = pickle.load(f)
        with open("./data/temp/img_x_test.pickle", mode='rb') as f:
           self.x_test = pickle.load(f)
        with open("./data/temp/img_y_test.pickle", mode='rb') as f:
           self.y_test = pickle.load(f)
        #self.x_train, self.x_test, self.y_train,self.y_test=XY
        nb_classes=len(category)
        #kerasで扱えるようにcategoriesをベクトルに変換
        self.y_train = to_categorical(self.y_train, nb_classes)
        self.y_test  = to_categorical(self.y_test, nb_classes)
        self.tryDataNum=0
    def createModel(self,num_layer, activation, mid_units, num_filters):
        """
        #モデル層を積み重ねる形式の記述方法 addで表記できるため便利
        model = models.Sequential()
        #畳み込み。(3,3)のカーネルを32種類使用。活性化関数はrelu。入力サイズ＝画像縦*横*RGB
        model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(250,250,3)))
        #「2×2」の大きさの最大プーリング層。入力画像内の「2×2」の領域で最大の数値を出力する。
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(64,(3,3),activation="relu"))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(128,(3,3),activation="relu"))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Conv2D(128,(3,3),activation="relu"))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Flatten())#平坦化（次元削減） – 1次元ベクトルに変換する。128*3*3=1152
        model.add(layers.Dropout(0.25))#過学習防止25%棄却　残り864
        model.add(layers.Dense(512,activation="relu"))#全結合層。出力512。ここの層を増やしてもいいかもしれない。
        model.add(layers.Dense(2,activation="sigmoid")) #分類先の種類分設定 2は識別種別数
        #モデル構成の確認
        
        model.summary()
        #二値交差エントロピー
        model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])
        """

        #引用 https://qiita.com/ryota717/items/28e2167ea69bee7e250d
        #num_layer : 畳込み層の数
        #activation : 活性化関数
        #mid_units : FC層のユニット数
        #num_filters : 各畳込み層のフィルタ数
        
        #今回は、パラメータをいじれるように、別の書式で記述

        inputs = Input((250,250,3))
        x = Convolution2D(filters=num_filters[0], kernel_size=(3,3), padding="same", activation=activation)(inputs)
        for i in range(1,num_layer):
            x = Convolution2D(filters=num_filters[i], kernel_size=(3,3), padding="same", activation=activation)(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(units=mid_units, activation=activation)(x)
        x = Dense(units=2, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=x)
        self.num_layer=num_layer
        self.activation=activation
        self.mid_units=mid_units
        self.num_filters=num_filters
        
        return model
        
        

    def objective(self,trial):#学習する目的関数の設定
        #セッションのクリア
        K.clear_session()

        #最適化するパラメータの設定
        #畳込み層の数
        num_layer = trial.suggest_int("num_layer", 1, 3)#1~3

        #FC(全結合)層のユニット数
        mid_units = int(trial.suggest_discrete_uniform("mid_units", 100, 500, 100))#第三引数はVBにおけるstep。間隔。

        #各畳込み層のフィルタ数
        num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 32, 128, 32)) for i in range(num_layer)]

        #活性化関数
        activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])

        #optimizer
        optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])
        self.optimizer=optimizer
        model = self.createModel(num_layer, activation, mid_units, num_filters)
        model.compile(optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"])#評価関数は正答率
        #学習が進まなくなったときに止める
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10)

        #学習実行
        #verbose ログ出力の指定。「0」だとログが出ないの設定。 epoch=学習する回数,batch_size=学習するデータサイズ validation_dataは検証用データ
        history = model.fit(self.x_train, self.y_train,  epochs=5, batch_size=6, validation_data=(self.x_test,self.y_test),callbacks=[early_stopping])
        self.history=history
        #modelの保存
        json_string = history.model.to_json()
        open('./data/param/model.json', 'w').write(json_string)
        #重みの保存

        hdf5_file = "./data/param/weight.hdf5"
        history.model.save_weights(hdf5_file)
        self.makeGraph()
        #検証用データに対する正答率が最大となるハイパーパラメータを求める
        return 1 - history.history["val_accuracy"][-1] #正答率最大→戻り値最小
    def makeGraph(self):#学習推移を出力
        #print(self.history.history.keys())

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(os.path.join("data/graph_accuracy/",f'accuracy_{self.tryDataNum}'))

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(os.path.join("data/graph_loss/", f'loss_{self.tryDataNum}.png'))
        #ログファイル書き出し
        path_w = f'data/log/log_{self.tryDataNum}.txt'

        crlf="\n"
        

        s = f'num_layer:{crlf}{self.num_layer}{crlf} activation:{crlf}{self.activation}{crlf} mid_units: {crlf} {self.mid_units}{crlf} num_filters:{crlf}{self.num_filters}{crlf} val_acc: {val_acc}{crlf} optimizer:{crlf}{self.optimizer}{crlf}'

        with open(path_w, mode='w') as f:
            f.write(s)
        self.tryDataNum+=1

    def optimizeModel(self):
        #studyオブジェクトの作成
        study = optuna.create_study()
        #最適化実行
        study.optimize(self.objective, n_trials=10)#試行回数
        #最適化したハイパーパラメータの確認←これが一番知りたかったやつ。
        print(study.best_params)
        print(study.best_params["activation"])
        #best param(dictionary形式)の保存
        with open("./data/param/bestparam.pickle", mode='wb') as f:
            pickle.dump(study.best_params,f)
        
        #最適化後の目的関数値 目的関数が最適化時どんな値(戻り値)になったか。
        print(study.best_value)
            
        #全試行の確認
        #print(study.trials)
    def learnWithOptimizedParam(self):
        with open("./data/param/bestparam.pickle", mode='rb') as f:#withはスコープを生成しない
            bestparam = pickle.load(f)
        
        print(bestparam)

        num_filters=[]
        for ss in bestparam.keys():
            if "num_filter" in ss:
                num_filters.append(int(bestparam[ss]))

        #ここら辺の説明は既出なので割愛
        model = self.createModel(bestparam["num_layer"], bestparam["activation"], bestparam["mid_units"], num_filters)
        model.compile(optimizer=bestparam["optimizer"],
                loss="binary_crossentropy",
                metrics=["accuracy"])
        
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10)
        history = model.fit(self.x_train, self.y_train,  epochs=5, batch_size=6, validation_data=(self.x_test,self.y_test),callbacks=[early_stopping])
        self.history=history
        #modelの保存
        
        json_string = history.model.to_json()
        open('./data/param/optimizedmodel.json', 'w').write(json_string)
        #重みの保存

        hdf5_file = "./data/param/optimizedModelWeight.hdf5"
        history.model.save_weights(hdf5_file)
        self.tryDataNum=-1
        self.makeGraph()




class TfDebug:
    def testGPURecognization(self):
        #TensorFlowがGPUを認識しているか確認
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
def main():
    #GPUが認識されているか確認
    #tfdebug=TfDebug()
    #tfdebug.testGPURecognization()
    
    #最適ハイパーパラメータの探索
    optimizeModel()

    #最適ハイパーパラメータを使用した学習モデルの構築
    #最適化探索が終わっていれば最適パラメータは保存されている。
    createOptimizedModel()

def optimizeModel():
    dataManager=DataManager()
    category= dataManager.arrangeData()
    category=dataManager.getCategory()
    learning=Learning(category)
    learning.optimizeModel()
def createOptimizedModel():
    dataManager=DataManager()
    category=dataManager.getCategory()
    learning=Learning(category)
    learning.learnWithOptimizedParam()

if __name__ == "__main__":
    main()


