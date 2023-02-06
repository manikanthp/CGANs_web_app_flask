from flask import Flask, render_template, request, redirect
from matplotlib.image import imsave
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)


app.config['IMAGE_UPLOADS'] = './static/images/uploads'
app.config['OUTPUT_IMAGE'] = './static/images/output'

generator_h5 = tf.keras.models.load_model('models/my_generator.h5')

@app.route("/", methods= ['GET', 'POST'])
def index():
    if request.method == "POST":
        if request.files:
            image = request.files['image']  # get file
            image.save(os.path.join(app.config['IMAGE_UPLOADS'], image.filename))
            #print('IMage saved')
            #pred = final(os.path.join(app.config['OUTPUT_IMAGE'], 'output.jpg'))
            img = load_img(path = os.path.join(app.config['IMAGE_UPLOADS'], image.filename), target_size = (256, 256))
            img = img_to_array(img)
            img = (img - 127.5)/127.5
            #reshape
            img_rows, img_cols, img_channels = img.shape
            image_array = img.reshape(1,img_rows, img_cols, img_channels)

            #generate map image
            predicted_image = generator_h5(image_array, training=True)[0]

            predicted_image = (predicted_image + 1) / 2.0
            predicted_image = np.array(predicted_image)
            #print(predicted_image)
            
            output_image  = os.path.join(app.config['OUTPUT_IMAGE'], image.filename)
            plt.imsave(output_image, predicted_image)

            output_path = f'/static/images/output/{image.filename}'
            input_path = f'/static/images/uploads/{image.filename}'
            #print('Predicted IMage saved')
            #print(output_image)

            return render_template('output.html',input_path = input_path, output_path = output_path)
            


    return render_template('base.html')




if __name__ == '__main__':
    app.run(debug=True)