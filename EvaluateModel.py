import tensorflow as tf
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='models/eflite0_b4_e65_threshold_0.65_gamma_1.25.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def run_inference(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output


# Evaluate the model using the test data
def evaluate_tflite_model(test_data, interpreter):
    num_correct = 0
    num_total = 0

    for image, label in test_data:
        # Preprocess the image and convert to the appropriate input shape
        image = np.expand_dims(image, axis=0).astype(np.float32)

        # Run inference
        output = run_inference(image)

        # Postprocess the output to get the predicted label
        # Assuming this is a classification task; adjust accordingly for detection.
        predicted_label = np.argmax(output)

        # Compare the predicted label with the true label
        if predicted_label == label:
            num_correct += 1
        num_total += 1

    accuracy = num_correct / num_total
    return accuracy


# Assuming 'test' is your test dataset
accuracy = evaluate_tflite_model('NEW DATA SPLIT/test', interpreter)
print(f"Re-evaluated accuracy of the TFLite model: {accuracy:.4f}")