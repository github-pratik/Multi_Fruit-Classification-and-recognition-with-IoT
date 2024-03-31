#save this as app.py
from flask import Flask, escape, request, render_template
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# model = load model("models/fruits.h5")
model = load_model("models/T1_SIG.h5")

#class_name  = ['OverripeApple','OverripeBanana','OverripeOrange','RipeApple','RipeBanana','RipeOrange','UnripeApple','UnripeBanana','UnripeOrange']

class_name = ['FrApple', 'RoApple','FOrange','ROrange','FBanana','UNanana','RbBanana','UNORange','UNApple']

#class_name = ['UnripeApple',
#              'RipeApple',
#              'OverripeApple',
#              'OverripeBanana',
#              'UnripeBanana',
#              'RipeBanana',
#              'UnripeOrange',
#              'OverripeOrange',
#              'RipeOrange']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    test_image = None 
    if request.method == 'POST':
        f = request.files['fruit']
        filename= f.filename
        #/Users/Pratik/GIT Clone/Fruit-Recognition/test
        target = os.path.join(APP_ROOT, 'image/')
        # print(target)
        des = "/".join([target, filename])
        f.save(des)

#        test_image= image.load_img(des,target_size=(300,300))
        test_image= image.load_img(des,target_size=(224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = model.predict(test_image)
        # print(prediction)

        predicted_class= class_name[np.argmax(prediction[0])]
        # print(predicted_class)
        confidence = round(np.max(prediction[0])*100)
        # print(confidence)

        return render_template("prediction.html", confidence= "chances -> "+str(confidence)+ "%", prediction = "prediction -> "+str(predicted_class))

    else:
        return render_template("prediction.html")




if __name__ == '__main__':
    app.debug = True
    app.run()