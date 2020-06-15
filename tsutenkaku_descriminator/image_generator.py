#�摜�̐�����

import os
import glob
import numpy as np
import learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# �摜���g������֐�
def draw_images(generator, x, dir_name, index):
    save_name = 'extened-' + str(index)
    g = generator.flow(x, batch_size=1, save_to_dir=output_dir,
                       save_prefix=save_name, save_format='jpeg')

    # 1�̓��͉摜���牽���g�����邩���w��i�����50���j
    for i in range(50):
        bach = g.next()


def run(category):
    # �o�͐�f�B���N�g���̐ݒ�
    output_dir = f'data_generated_{category}'

    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    # �g������摜�̓ǂݍ���
    images = glob.glob(os.path.join(f'data/{category}', "*.jpg"))#�t�@�C�����O��/�͂���Ȃ��炵��

    # ImageDataGenerator���`
    datagen = ImageDataGenerator(rotation_range=30,
                                width_shift_range=20,
                                height_shift_range=0.,
                                zoom_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True)
    # �ǂݍ��񂾉摜�����Ɋg��
    for i in range(len(images)):
        img = load_img(images[i])
        img = img.resize((150, 150))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        draw_images(datagen, x, output_dir, i)

def main():
    category=learning.DataManager.getCategory
    for cat in category:
        run(cat)
    
if __name__ == "__main__":
    main()