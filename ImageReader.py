import tensorflow as tf
import pathlib
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import WindowsPath

print(tf.__version__)

data_dir = WindowsPath("C:/TFG/PyProjects/DataSet/rscbjbr9sj-2/ChestXRay2017/chest_xray/train/PNEUMONIA")
image_count = len(list(data_dir.glob('*/*.jpeg')))
print(image_count)

data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

image_count = len(list(data_dir.glob('*/*.jpg')))
image_count

print(image_count)

roses = list(data_dir.glob('roses/*'))

for image_path in roses[:3]:
    print(str(image_path))
    '''display.display(Image.open(str(image_path)))
    img = Image.open(str(image_path))
    img.show()'''
    img = mpimg.imread(str(image_path))

    plt.imshow(img)
    plt.show()
