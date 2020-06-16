#綾鷹を選ばせるプログラム

from tensorflow.keras import models
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
import learning

class Prediction:
    def predictTsutenkaku(self):

        #保存したモデルの読み込み
        model = model_from_json(open('./data/param/optimizedModel.json').read())
        #保存した重みの読み込み
        model.load_weights('./data/param/optimizedModelWeight.hdf5')

        categories = ["通天閣","太陽の塔"]

        #画像を読み込む
        img_path = "./data/predictImg/face.jpg"
        img = image.load_img(img_path,target_size=(250, 250, 3))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        #予測
        features = model.predict(x)

        #予測結果によって処理を分ける
        if(features[0,0]>features[0,1]):
            print(f'{categories[0]}である確率が{features[0,0]}%、{categories[1]}である確率が{features[0,1]}%で、あなたの顔はどちらかというと{categories[0]}に似ています。')
        else:
            print(f'{categories[0]}である確率が{features[0,0]}%、{categories[1]}である確率が{features[0,1]}%で、あなたの顔はどちらかというと{categories[1]}に似ています。')



def main():
    prediction=Prediction()
    prediction.predictTsutenkaku()
    pass
if __name__ == "__main__":
    main()