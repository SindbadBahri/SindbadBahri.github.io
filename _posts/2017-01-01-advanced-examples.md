```python
import os
import glob
import random
import shutil
import itertools
import warnings
import matplotlib.pyplot as plt 
%matplotlib inline
###############
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
```


```python
os.chdir('c:\\Users\\rasou\\Desktop\\DeepLearning\\dogs-vs-cats')
os.getcwd()
```




    'c:\\Users\\rasou\\Desktop\\DeepLearning\\dogs-vs-cats'




```python
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for i in random.sample(glob.glob('cat*'), 500):
        shutil.move(i, 'train/cat')      
    for i in random.sample(glob.glob('dog*'), 500):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 100):
        shutil.move(i, 'valid/cat')        
    for i in random.sample(glob.glob('dog*'), 100):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 50):
        shutil.move(i, 'test/cat')      
    for i in random.sample(glob.glob('dog*'), 50):
        shutil.move(i, 'test/dog')

```


```python
os.chdir('c:\\Users\\rasou\\Desktop\\DeepLearning')
os.getcwd()
```




    'c:\\Users\\rasou\\Desktop\\DeepLearning'




```python
train_path = 'c:\\Users\\rasou\\Desktop\\DeepLearning\\dogs-vs-cats\\train'
valid_path = 'c:\\Users\\rasou\\Desktop\\DeepLearning\\dogs-vs-cats\\valid'
test_path = 'c:\\Users\\rasou\\Desktop\\DeepLearning\\dogs-vs-cats\\test'
```


```python
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)
```

    Found 1000 images belonging to 2 classes.
    Found 200 images belonging to 2 classes.
    Found 100 images belonging to 2 classes.
    


```python
imgs, labels = next(train_batches)
labels
```




    array([[0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [1., 0.]], dtype=float32)




```python
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```


```python
plotImages(imgs)
print(labels)
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    
![svg](output_8_1.svg)
    


    [[0. 1.]
     [0. 1.]
     [0. 1.]
     [0. 1.]
     [0. 1.]
     [0. 1.]
     [0. 1.]
     [1. 0.]
     [1. 0.]
     [1. 0.]]
    


```python
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])
```


```python
model.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_6 (Conv2D)            (None, 224, 224, 32)      896       
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 112, 112, 32)      0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 112, 112, 64)      18496     
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 56, 56, 64)        0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 200704)            0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 2)                 401410    
    =================================================================
    Total params: 420,802
    Trainable params: 420,802
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=10,
    verbose=2
)
```

    Epoch 1/10
    100/100 - 85s - loss: 20.4015 - accuracy: 0.5750 - val_loss: 8.9690 - val_accuracy: 0.5850
    Epoch 2/10
    100/100 - 71s - loss: 4.3070 - accuracy: 0.7230 - val_loss: 6.6126 - val_accuracy: 0.5850
    Epoch 3/10
    100/100 - 79s - loss: 0.9511 - accuracy: 0.8620 - val_loss: 3.5938 - val_accuracy: 0.5900
    Epoch 4/10
    100/100 - 69s - loss: 0.1879 - accuracy: 0.9550 - val_loss: 4.2396 - val_accuracy: 0.5900
    Epoch 5/10
    100/100 - 87s - loss: 0.0951 - accuracy: 0.9740 - val_loss: 4.1277 - val_accuracy: 0.5850
    Epoch 6/10
    100/100 - 69s - loss: 0.0852 - accuracy: 0.9800 - val_loss: 3.7134 - val_accuracy: 0.5800
    Epoch 7/10
    100/100 - 76s - loss: 0.0256 - accuracy: 0.9920 - val_loss: 3.9800 - val_accuracy: 0.5750
    Epoch 8/10
    100/100 - 74s - loss: 0.0145 - accuracy: 0.9940 - val_loss: 4.2553 - val_accuracy: 0.5850
    Epoch 9/10
    100/100 - 73s - loss: 0.0038 - accuracy: 0.9980 - val_loss: 3.7967 - val_accuracy: 0.5800
    Epoch 10/10
    100/100 - 76s - loss: 0.0015 - accuracy: 1.0000 - val_loss: 3.6469 - val_accuracy: 0.5800
    




    <tensorflow.python.keras.callbacks.History at 0x1af99286580>




```python
test_imgs, test_labels= next(test_batches)
```


```python

predictions= model.predict(x=test_batches, steps= len(test_batches), verbose=0)
```


```python
cm= confusion_matrix(y_true= test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
```


```python
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```


```python
cm_plot_labels=['cat', 'dog']
```


```python
plot_confusion_matrix(cm, cm_plot_labels,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues)
```

    Confusion matrix, without normalization
    [[28 22]
     [13 37]]
    


    
![svg](output_18_1.svg)
    



```python
vgg16_model= tf.keras.applications.vgg16.VGG16()
```


```python
vgg16_model.summary()
```

    Model: "vgg16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_4 (InputLayer)         [(None, 224, 224, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    fc1 (Dense)                  (None, 4096)              102764544 
    _________________________________________________________________
    fc2 (Dense)                  (None, 4096)              16781312  
    _________________________________________________________________
    predictions (Dense)          (None, 1000)              4097000   
    =================================================================
    Total params: 138,357,544
    Trainable params: 138,357,544
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model =  Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
```


```python
model.summary()
```

    Model: "sequential_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    fc1 (Dense)                  (None, 4096)              102764544 
    _________________________________________________________________
    fc2 (Dense)                  (None, 4096)              16781312  
    =================================================================
    Total params: 134,260,544
    Trainable params: 134,260,544
    Non-trainable params: 0
    _________________________________________________________________
    


```python
for layer in model.layers:
    layer.trainable=False
```


```python
model.add(Dense(units=2, activation='softmax'))
```


```python
model.summary()
```

    Model: "sequential_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    fc1 (Dense)                  (None, 4096)              102764544 
    _________________________________________________________________
    fc2 (Dense)                  (None, 4096)              16781312  
    _________________________________________________________________
    dense_8 (Dense)              (None, 2)                 8194      
    =================================================================
    Total params: 134,268,738
    Trainable params: 8,194
    Non-trainable params: 134,260,544
    _________________________________________________________________
    


```python
model.compile(optimizer= Adam(learning_rate=0.0001), loss= 'categorical_crossentropy', metrics=['accuracy'])
```


```python
model.fit(x=train_batches, validation_data= valid_batches,steps_per_epoch=len(train_batches),validation_steps=len(valid_batches), epochs=5, verbose=2)
```

    Epoch 1/5
    100/100 - 722s - loss: 0.1222 - accuracy: 0.9470 - val_loss: 0.0830 - val_accuracy: 0.9650
    Epoch 2/5
    100/100 - 698s - loss: 0.0572 - accuracy: 0.9780 - val_loss: 0.0707 - val_accuracy: 0.9700
    Epoch 3/5
    100/100 - 499s - loss: 0.0366 - accuracy: 0.9850 - val_loss: 0.0655 - val_accuracy: 0.9700
    Epoch 4/5
    100/100 - 515s - loss: 0.0254 - accuracy: 0.9910 - val_loss: 0.0669 - val_accuracy: 0.9750
    Epoch 5/5
    100/100 - 579s - loss: 0.0185 - accuracy: 0.9960 - val_loss: 0.0663 - val_accuracy: 0.9800
    




    <tensorflow.python.keras.callbacks.History at 0x1af9e2d7910>




```python
test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)
print(test_labels)
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    
![svg](output_28_1.svg)
    


    [[1. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]
     [1. 0.]]
    


```python
predictions= model.predict(x= test_batches, steps= len(test_batches), verbose=0)
```


```python
cm= confusion_matrix(y_true= test_batches.classes, y_pred= np.argmax(predictions, axis=-1))
```


```python
cm_plot_labels=['cat', 'dog']
plot_confusion_matrix(cm, cm_plot_labels,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues)
```

    Confusion matrix, without normalization
    [[48  2]
     [ 0 50]]
    


    
![svg](output_31_1.svg)
    

