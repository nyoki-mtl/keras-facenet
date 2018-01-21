# keras-facenet
Facenet implementation by Keras2.0

## Pretrained model
You can quickly start facenet with pretrained Keras model (trained by MS-Celeb-1M dataset).
- Download model from [here](https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn)
- If you use Ubuntu

```
from keras.models import load_model
model_path = '../model/keras/model/facenet_keras.h5'
model = load_model(model_path)
```

- If you use Windows

```
import sys
sys.path.append('../code/')
from inception_resnet_v1 import *
model = InceptionResNetV1(weights_path='../model/keras/weights/facenet_keras_weights.h5')
```

You can also create Keras model from pretrained tensorflow model.
- Download model from [here](https://github.com/davidsandberg/facenet) and save it in model/tf/
- Convert model for Keras in [tf_to_keras.ipynb](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb)


## Demo
- [Face vector calculation](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/demo-images.ipynb)
- [Classification with SVM](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/demo-svm.ipynb)
- [Web camera demo](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/demo-webcam.ipynb)

## Environments
Ubuntu16.04 or Windows10  
python3.6.2  
tensorflow: 1.3.0  
keras: 2.1.2  
