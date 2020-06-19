# coding: utf-8
#画像の分配、移動、水増し

import os
import glob
import numpy as np
import learning
import shutil
import random, math
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# 画像を拡張する関数
def draw_images(generator, x, dir_name, index):
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=dir_name,
                       save_prefix=save_name, save_format='jpeg')

    # 1つの入力画像から何枚拡張するかを指定（今回は50枚）
    for i in range(50):
        bach = g.next()


def run(category):
    allpic=[]
    input_dir=f'./data/{category}'
    # 出力先ディレクトリの設定
    output_dir = f'./data/data_generated_{category}'
    moveto_dir=f'./data/data_valid1_{category}'

    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    if not(os.path.exists(moveto_dir)):
        os.mkdir(moveto_dir)


    #テストデータの分離移動
    files = os.listdir(input_dir)
    files_file = [f for f in files if os.path.isfile(os.path.join(input_dir, f))]#ディレクトリ下のファイルのみ全取得
    for i in files_file:
        allpic.append(i)#ファイル名が入る

    random.shuffle(allpic)
    th = math.floor(len(allpic) * 0.5)
    trainImg=allpic[0:th]
    testImg=allpic[th:]

    for i in testImg:
        shutil.move(os.path.join(input_dir,i), moveto_dir)#ディレクトリ移動

    # 拡張する画像の読み込み
    images = glob.glob(os.path.join(input_dir, "*.jpeg"))#ファイル直前の/はいらないらしい

    # ImageDataGeneratorを定義
    datagen = ImageDataGenerator(rotation_range=30,
                                width_shift_range=20,
                                height_shift_range=0.,
                                zoom_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True)
    
    # 読み込んだ画像を順に拡張
    for i in range(len(images)):
        img = load_img(images[i])
        img = img.resize((250, 250))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        draw_images(datagen, x, output_dir, i)


def main():
    dataManager=learning.DataManager()
    category=dataManager.getCategory()
    for cat in category:
        run(cat)
    
if __name__ == "__main__":
    main()