import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import QuantizationConfig

# Load the existing TFLite model
interpreter = tf.lite.Interpreter(model_path='old_models/efficientdet_lite0_whole_b2_e65_with_augmented_data.tflite')
interpreter.allocate_tensors()

# Get the input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create the QuantizationConfig
quantization_config = QuantizationConfig(
    inference_input_type=tf.float32,
    inference_output_type=tf.float32
)

# Convert the model to TFLite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(interpreter)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = your_representative_dataset_generator
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = quantization_config.inference_input_type
converter.inference_output_type = quantization_config.inference_output_type

tflite_model = converter.convert()

# Save the quantized model
with open('models/quantized_model.tflite', 'wb') as f:
    f.write(tflite_model)