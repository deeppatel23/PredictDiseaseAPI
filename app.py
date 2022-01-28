import io
import flask
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file, jsonify
from tensorflow.keras.preprocessing.image import load_img , img_to_array

app = Flask(__name__)
model = load_model('model.h5')


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['Acne and Rosacea' ,'Atopic Dermatitis', 'Bullous Disease' , 'Eczema' , 'Exanthems and Drug Eruptions' ,'Lupus and other Connective Tissue diseases' ,'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease' ,'Poison Ivy Photos and other Contact Dermatitis' ,'Vascular Tumor']

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    return img

def predict(imageFile , model):
    img_bytes = imageFile.read()
    prepared_image = prepare_image(img_bytes)
    img = img_to_array(prepared_image)
    img = img.reshape(1 , 224 ,224 ,3)

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(10):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result




@app.route('/')
def home():
        return render_template("index.html")

@app.route('/predict' , methods = ['GET' , 'POST'])
def result():
    error = ''
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            class_result , prob_result = predict(file , model)

            predictions = {
                    "class1":class_result[0],
                    "class2":class_result[1],
                    "class3":class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }

        else:
            error = "Please upload images of jpg , jpeg and png extension only"

        if(len(error) == 0):
            return  jsonify(predictions)
        else:
            return render_template('index.html' , error = error)            
    else:
        return render_template('index.html')

