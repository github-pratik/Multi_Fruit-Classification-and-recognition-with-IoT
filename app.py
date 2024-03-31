
from flask import Flask, escape, request, render_template
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageOps

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# model = load model("models/fruits.h5")
#model = load_model("models/T1_SIG.h5")
interpreter = tf.lite.Interpreter(model_path="models/Lite_Model/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['RipeApple', 'OverripeApple', 'RipeOrange', 'OverripeOrange', 'RipeBanana', 'OverripeBanana', 'UnripeBanana', 'UnripeORange', 'UnripeApple']
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        f = request.files['fruit']
        filename = f.filename
        target = os.path.join(APP_ROOT, 'image/')
        des = os.path.join(target, filename)
        f.save(des)

        # For debugging, print the filename to check if the file is received
        print("Received file:", filename)

        # Resize and preprocess the uploaded image
        image = Image.open(des).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # For debugging, print the image data to check if preprocessing is correct
        print("Preprocessed image data:", data)

        # Perform model inference
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # For debugging, print the output data to check if inference is correct
        #print("Output data:", output_data)

        # Process output
        threshold = 0.2  # Set threshold to 0.2
        top_indices = np.where(output_data[0] > threshold)[0]
        top_indices = top_indices[np.argsort(output_data[0][top_indices])[::-1]][:3]
        top_predictions = [(class_names[i], round(output_data[0][i] * 100, 2)) for i in top_indices]

        # Sort the top predictions by confidence score in descending order
        top_predictions.sort(key=lambda x: x[1], reverse=True)

        top_3_predictions = top_predictions[:3]
        print("Top predictions:", top_predictions)

        return render_template("prediction.html", top_predictions=top_3_predictions)
    else:
        return render_template("prediction.html")


if __name__ == '__main__':
    app.debug = True
    app.run()