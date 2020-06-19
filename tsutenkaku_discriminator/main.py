#綾鷹を選ばせるプログラム

from tensorflow.keras import models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.compat.v1.keras.utils import to_categorical
import numpy as np
import learning
import os, glob
from PIL import Image
import pickle

class Prediction:
    def predictTsutenkaku(self):

        #保存したモデルの読み込み
        model = model_from_json(open('./data/param/optimizedModel.json').read())
        with open("./data/param/bestparam.pickle", mode='rb') as f:
            bestparam = pickle.load(f)
        model.compile(optimizer=bestparam["optimizer"],
                loss="binary_crossentropy",
                metrics=["accuracy"])
        #保存した重みの読み込み
        model.load_weights('./data/param/optimizedModelWeight.hdf5')
        model.summary()
        
        categories = ["通天閣","太陽の塔"]

        #画像を読み込む
        img_path = "./data/predictImg/face1.jpg"
        files = glob.glob(img_path) #ファイルの取得


        img = Image.open(img_path)
        img = img.convert("RGB")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img=self.expand2square(img, (0, 0, 0)).resize((250, 250), Image.LANCZOS)
        x=np.asarray(img)
        x=np.array(x).astype(np.float)/255
        x = np.expand_dims(x, axis=0)
        """
        y=[0]
        y=to_categorical(y,2)
        score = model.evaluate(x=x,y=y)

        print('loss=', score[0])
        print('accuracy=', score[1])
        """

        #予測
        features = model.predict(x)

        #予測結果によって処理を分ける
        if(features[0,0]>features[0,1]):
            print(f'{categories[0]}である確率が{features[0,0]}%、{categories[1]}である確率が{features[0,1]}%で、あなたの顔はどちらかというと{categories[0]}に似ています。')
        else:
            print(f'{categories[0]}である確率が{features[0,0]}%、{categories[1]}である確率が{features[0,1]}%で、あなたの顔はどちらかというと{categories[1]}に似ています。')

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


def main():
    prediction=Prediction()
    prediction.predictTsutenkaku()
    pass
if __name__ == "__main__":
    main()