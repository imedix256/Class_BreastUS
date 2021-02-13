import sys
print(sys.version)
print(sys.path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

import os, cv2, zipfile, io, re, glob
from PIL import Image
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#from keras.applications.vgg16 import VGG16
#from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.xception import Xception
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Activation, Dropout
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

#データ取得
# ZIP読み込み
#z = zipfile.ZipFile('/home/imedix/amed18_2/hikakudata.zip')
z = zipfile.ZipFile('/home/imedix/AMED_PGGAN_CLASS/Real/hikakudata.zip')

#img_dirs = [ x for x in z.namelist() if re.search("^hikakudata/Training/.*/$", x)]
img_dirs = [ x for x in z.namelist() if re.search("^hikakudata/Train/.*/$", x)]
print (img_dirs)
# 不要な文字列削除
#img_dirs = [ x.replace('hikakudata/Training/', '') for x in img_dirs]
img_dirs = [ x.replace('hikakudata/Train/', '') for x in img_dirs]
img_dirs = [ x.replace('/', '') for x in img_dirs]
img_dirs.sort()

print (img_dirs)

classes = img_dirs
#classes = ['cyst','hcc','hemangioma','meta']

num_classes = len(classes)

del img_dirs

# 画像サイズ
#image_size = 150
image_size = 256

# 画像を取得し、配列に変換
def im2array(path):
    X = []
    y = []
    class_num = 0

    for class_name in classes:
        if class_num == num_classes : break
        imgfiles = [ x for x in z.namelist() if re.search("^" + path + class_name + "/.*bmp$", x)] 
        for imgfile in imgfiles:
            # ZIPから画像読み込み
            image = Image.open(io.BytesIO(z.read(imgfile)))
            # RGB変換
            image = image.convert('RGB')
            # リサイズ
            image = image.resize((image_size, image_size))
            # 画像から配列に変換
            data = np.asarray(image)
            X.append(data)
            y.append(classes.index(class_name))
        class_num += 1

    X = np.array(X)
    y = np.array(y)

    return X, y

#trainデータ取得
#X_train, y_train = im2array("hikakudata/Training/")
X_train, y_train = im2array("hikakudata/Train/")
print(X_train.shape, y_train.shape)

#testデータ取得
#X_test, y_test = im2array("hikakudata/Test/")
#print(X_test.shape, y_test.shape)

#del z
X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')

# 正規化
X_train /= 255
#X_test /= 255

# one-hot 変換
y_train = to_categorical(y_train, num_classes = num_classes)
#y_test = to_categorical(y_test, num_classes = num_classes)
#print(y_train.shape, y_test.shape)

#trainデータからvalidデータを分割
#X_train, X_valid, y_train, y_valid = train_test_split(
#    X_train,
#    y_train,
#    random_state = 0,
#    stratify = y_train,
#    test_size = 0.2
#)
#print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape) 

#trainデータからvalidデータを分割 klearn.model_selection import KFold を使う
kf = KFold(n_splits=10, shuffle=True)

for train_index, val_index in kf.split(X_train,y_train):

    X_tra=X_train[train_index]
    y_tra=y_train[train_index]
    X_val=X_train[val_index]
    y_val=y_train[val_index]

#Data Augmentation
datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = False
)

#Callback
# EarlyStopping
early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    verbose = 1
)

# ModelCheckpoint
weights_dir = './weights/'
if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
model_checkpoint = ModelCheckpoint(
    weights_dir + "val_loss{val_loss:.3f}.hdf5",
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = True,
    period = 3
)

# reduce learning rate
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.1,
    patience = 3,
    verbose = 1
)

# log for TensorBoard
logging = TensorBoard(log_dir = "log/")

#各種関数定義
# モデル学習
def model_fit():
    hist = model.fit_generator(
        datagen.flow(X_train, y_train, batch_size = 32),
        steps_per_epoch = X_train.shape[0] // 32,
        epochs = 50,
        validation_data = (X_val, y_val),
        callbacks = [early_stopping, reduce_lr],
        shuffle = True,
        verbose = 1
    )
    return hist

#結果テキスト保存
nb_epoch = 50

result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def save_history(hist, result_file):
    loss_v = hist.history['loss']
    acc_v = hist.history['acc']
    val_loss_v = hist.history['val_loss']
    val_acc_v = hist.history['val_acc']
    nb_epoch = len(acc_v)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss_v[i], acc_v[i], val_loss_v[i], val_acc_v[i]))

# モデル保存
model_dir = './model/'
weights_dir = './weights/'
if os.path.exists(model_dir) == False : os.mkdir(model_dir)

#def model_save(model_name):
#    model.save(model_dir + 'model_' + model_name + '.hdf5')

    # optimizerのない軽量モデルを保存（学習や評価不可だが、予測は可能）
#    model.save(model_dir + 'model_' + model_name + '-opt.hdf5', include_optimizer = False)

def model_save(model_name):
    model_json = model.to_json()
    with open(model_dir +"BreastUS_original_cross" + model_name + ".json", "w") as json_file:
     json_file.write(model_json)
    model.save(model_dir + 'model_' + model_name + '.hdf5')
    model.save_weights(weights_dir + 'BreastUS_original_cross' + model_name + '.h5')
    # optimizerのない軽量モデルを保存（学習や評価不可だが、予測は可能）
    model.save(model_dir + 'model_' + model_name + '-opt.hdf5', include_optimizer = False)

# 学習曲線をプロット
def learning_plot(title):
    plt.figure(figsize = (18,6))

    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["acc"], label = "acc", marker = "o")
    plt.plot(hist.history["val_acc"], label = "val_acc", marker = "o")
    #plt.yticks(np.arange())
    #plt.xticks(np.arange())
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = 'gray', alpha = 0.2)

    # loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label = "loss", marker = "o")
    plt.plot(hist.history["val_loss"], label = "val_loss", marker = "o")
    #plt.yticks(np.arange())
    #plt.xticks(np.arange())
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = 'gray', alpha = 0.2)

    plt.show()

# モデル評価
def model_evaluate():
   #score = model.evaluate(X_test, y_test, verbose = 1)
    score = model.evaluate(X_val, y_val, verbose = 1)
    print("evaluate loss: {[0]:.4f}".format(score))
    print("evaluate acc: {[1]:.1%}".format(score))
    dt_now = datetime.datetime.now()
    fileobj = open("InceptionvResNet_V2_775results.txt", "w")
#sample.txtを書き出しモードで開きます。
    fileobj.write('{0:%Y_%m_%d_%H_%M_%S}'.format(dt_now) + "evaluate loss: {[0]:.4f}".format(score) + ' ' + "evaluate acc: {[1]:.1%}".format(score))
    fileobj.close()

#Inception_V3
base_model = InceptionResNetV2(
    include_top = False,
    weights = "imagenet",
    input_shape = None
)

# 全結合層の新規構築
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

# ネットワーク定義
model = Model(inputs = base_model.input, outputs = predictions)
print("{}層".format(len(model.layers)))

# 775層までfreeze
for layer in model.layers[:775]:
    layer.trainable = False

    # Batch Normalization の freeze解除
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True

#775層以降、学習させる
for layer in model.layers[775:]:
    layer.trainable = True

# layer.trainableの設定後にcompile
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['acc']
)

hist=model_fit()
learning_plot("InceptionResNetV2_775_256")
model_evaluate()
model_save("InceptionResNetV2_775_256")
save_history(hist, os.path.join(result_dir, 'history_inceptionresnetv2_775_256.txt'))
