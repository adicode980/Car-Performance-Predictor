import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
from sklearn.externals import joblib


app = Flask(__name__)
model1 = pickle.load(open("model1.pkl","rb"))
model2 = pickle.load(open("model2.pkl","rb"))
model3 = pickle.load(open("model3.pkl","rb"))
transformer = joblib.load("transformer.pkl")
transformer2 = joblib.load("transformer2.pkl")
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predictmpl",methods = ['POST'])
def predictmpl():

    int_features = [str(x) for x in request.form.values()]
    int_features[1] = float(int_features[1])
    int_features[2] = float(int_features[2])
    final_features = [np.array(int_features)]
    final_features = transformer.transform(final_features)
    prediction = model1.predict(final_features)
    output = round(prediction[0] ,2)
    return render_template('index.html', Mileage= "The Mileage  should be {} kmpl".format(output))
@app.route("/predicthp",methods = ['POST'])
def predicthp():
    int_features = [str(x) for x in request.form.values()]
    int_features[1] = float(int_features[1])
    int_features[2] = float(int_features[2])
    final_features = [np.array(int_features)]
    final_features = transformer.transform(final_features)
    prediction = model2.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', Horsepower="The Horsepower  should be {} hp".format(output))

@app.route("/predictprice",methods = ["POST"])
def predictprice():
    int_features = [str(x) for x in request.form.values()]
    int_features[1] = float(int_features[1])
    int_features[2] = float(int_features[2])
    final_features = [np.array(int_features)]
    final_features = transformer2.transform(final_features)
    prediction = model3.predict(final_features)
    output = round(prediction[0])
    return render_template('index.html', Price = "The Average Price of car with choosen features should be {} rupees".format(output))


@app.route('/predict_api',methods = ["POST"])
def predict_api():

    data = request.get_json(force = True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ =="__main__":
    app.run(debug = True)