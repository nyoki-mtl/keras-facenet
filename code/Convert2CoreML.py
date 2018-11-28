#reference https://gist.github.com/viteinfinite/1901d9cb6d26c21967a08aa5534e05c2#file-convert-py
#input and outputs are scaled according to https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/demo-webcam.ipynb
#input is normalized per image through average and standard deviation 
#output is done through L2 normalization

import copy
import coremltools
from coremltools.proto import NeuralNetwork_pb2
from coremltools.models.neural_network.quantization_utils import *
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model
import os.path
import sys


#sys.path.append('/code/') # Import "code" from https://github.com/nyoki-mtl/keras-facenet 

from inception_resnet_v1 import *

#model source https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn

model = InceptionResNetV1(weights_path='facenet_keras_weights.h5')

def convert_lambda(layer):
    if layer.function == scaling:
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "scaling"
        params.parameters["scale"].doubleValue = layer.arguments['scale']
        return params
    else:
        return None

coreml_model = coremltools.converters.keras.convert(
    model,
    input_names="image",
    image_input_names="image",
    output_names="output",
    add_custom_layers=True,
    model_precision='float16',
    custom_conversion_functions={ "Lambda": convert_lambda })


spec = coreml_model.get_spec()

# get NN portion of the spec
nn_spec = spec.neuralNetwork
layers = nn_spec.layers # this is a list of all the layers
layers_copy = copy.deepcopy(layers) # make a copy of the layers, these will be added back later
del nn_spec.layers[:] # delete all the layers

# add a scale layer now
# since mlmodel is in protobuf format, we can add proto messages directly
# To look at more examples on how to add other layers: see "builder.py" file in coremltools repo
image_norm_layer = nn_spec.layers.add()
image_norm_layer.name = 'image_norm_layer'
image_norm_layer.input.append('image')
image_norm_layer.output.append('image_normalized')

params = image_norm_layer.mvn
params.acrossChannels = False
params.normalizeVariance = True
params.epsilon = 1e-6

# now add back the rest of the layers (which happens to be just one in this case: the crop layer)
nn_spec.layers.extend(layers_copy)

# need to also change the input of the crop layer to match the output of the scale layer
nn_spec.layers[1].input[0] = 'image_normalized'

nn_spec.layers[len(nn_spec.layers)-1].output[0] = 'embeddings_raw'

embeddings_norm_layer = nn_spec.layers.add()
embeddings_norm_layer.name = 'embeddings_norm_layer'
embeddings_norm_layer.input.append('embeddings_raw')
embeddings_norm_layer.output.append('output')

params =  embeddings_norm_layer.l2normalize
params.epsilon = 1e-6
#nn_spec.layers[len(nn_spec.layers)-2].input[0] = 'embeddings'





coreml_model = coremltools.models.MLModel(spec)

spec = coreml_model.get_spec()

coremltools.utils.save_spec(spec, 'facenet_keras_weights_coreml.mlmodel')
