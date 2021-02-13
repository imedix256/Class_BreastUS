from __future__ import print_function

import sys
print(sys.version)
print(sys.path)
import struct
import numpy as np
import os, cv2, zipfile, io, re, glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from PIL import Image
from keras.models import model_from_json
from keras.models import Model, load_model
from keras.layers.core import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

##データ取得
## ZIP読み込み
z = zipfile.ZipFile('/home/imedix/AMED_PGGAN_CLASS/Real/hikakudata.zip')


img_dirs = [ x for x in z.namelist() if re.search("^hikakudata/Test/.*/$", x)]
print (img_dirs)
## 不要な文字列削除
img_dirs = [ x.replace('hikakudata/Test/', '') for x in img_dirs]
img_dirs = [ x.replace('/', '') for x in img_dirs]
img_dirs.sort()

print (img_dirs)

classes = img_dirs

num_classes = len(classes)

del img_dirs

## 画像サイズ
image_size = 256

## 画像を取得し、配列に変換
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

##testデータ取得
X_test, y_test = im2array("hikakudata/Test/")
print(X_test.shape, y_test.shape)

## データ型の変換
X_test = X_test.astype('float32')

## 正規化
X_test /= 255

## one-hot 変換
y_test = to_categorical(y_test, num_classes = num_classes)
print(y_test.shape)


## InceptionResNetV2学習済みモデルと重みをロード
model_dir = './model/'
weights_dir = './weights/'
    ## Fully-connected層（FC）はいらないのでinclude_top=False）
#    input_tensor = Input(shape=(img_rows, img_cols, 3))
##   vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
f = open(model_dir +"BreastUS_original_crossInceptionResNetV2_775_256.json", 'r')
loaded_model_json = f.read()
f.close()
model = model_from_json(loaded_model_json)

#model = model_from_json(open('chest_finetunig2.json').read())
model.load_weights(weights_dir + 'BreastUS_original_crossInceptionResNetV2_775_256.h5')


#評価　一覧
y_pred = model.predict(X_test, verbose=1)
y_pred_keras = model.predict(X_test, verbose=1)

# testデータ n件の正解ラベル
#true_classes = np.argmax(y_test[0:n], axis = 1)
true_classes = np.argmax(y_test[0:300], axis = 1)
print('correct:', true_classes)
np.savetxt('correctIncResNetV2_775_256_real.csv',true_classes,delimiter=',')

# testデータ n件の予測ラベル
#pred_classes = np.argmax(model.predict(X_test[0:n]), axis = 1)
pred_classes = np.argmax(model.predict(X_test[0:300]), axis = 1)
print('prediction:', pred_classes)
np.savetxt('predictionsIncResNetV2_775_256_real.csv',pred_classes,delimiter=',')


print('2x2_IncRenNetV2_775_256')
print(confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1)))
print(classification_report(np.argmax(y_test, 1), np.argmax(y_pred, 1)))
# ROC Curve
#y_pred_keras = model.predict(X_test, verbose=1).ravel()
y_pred_keras = model.predict(X_test, verbose=1)

#print('y_test:', y_test[:,0])
#print('y_pred_keras:', y_pred_keras[:,0])
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test[:,0], y_pred_keras[:,0], drop_intermediate=False)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test[:,0], y_pred_keras[:,0])
#print('FPR:', fpr_keras)
#print('TPR:', tpr_keras)

plt.plot(fpr_keras, tpr_keras, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.savefig('/home/imedix/AMED_PGGAN_CLASS/Real/Real_BreastUS_IRNV2_775_256_Test_roc_curve.png')


# AUC
auc_keras = auc(fpr_keras, tpr_keras)
print('AUC:', auc_keras)



