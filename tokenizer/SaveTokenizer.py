# from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import TFBertForPreTraining  # Or BertForPreTraining for loading pretraining heads
# from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
# model = AutoModelForPreTraining.from_pretrained('adalbertojunior/distilbert-portuguese-cased')
model = TFBertForPreTraining.from_pretrained('adalbertojunior/distilbert-portuguese-cased')

import tensorflow as tf
from transformers import TFDistilBertPreTrainedModel

input_spec = tf.TensorSpec([1, 384], tf.int32)
model._set_inputs(input_spec, training=False)

print(model.inputs)
print(model.outputs)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# For normal conversion:
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
# For conversion with FP16 quantization:
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.float16]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True

tflite_model = converter.convert()

open("distilbert-squad-384.tflite", "wb").write(tflite_model)