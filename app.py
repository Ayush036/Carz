from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import os
import datetime
import pickle
import sklearn
webapp_root = "webapp"
template_dir = os.path.join(webapp_root, "templates")



app = Flask(__name__, template_folder=template_dir)
app.static_folder=os.path.join(webapp_root, "static")

#app.static_folder="static"

#infile = open("rain_model1.pkl",'rb')
#new_dict = pickle.load(infile)
model = pickle.load(open("car_model.sav", 'rb'))
print("Model Loaded")

# @app.route("/",methods=['GET'])
# @cross_origin()
# def home():
# 	return render_template("index.html")
#
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
@cross_origin()
def predict():
	if request.method == "POST":
		brand=float(request.form['brand'])
		year=float(request.form['year'])
		mileage=float(request.form['mileage'])
		engine_size=float(request.form['Engine Size'])
		transmission=float(request.form['transmission'])

		if transmission==1:
			transmission_Manual=1
			transmission_Other=0
			transmission_Semi_Auto=0
		elif transmission==4:
			transmission_Manual=0
			transmission_Other=1
			transmission_Semi_Auto=0
		elif transmission==2:
			transmission_Manual=0
			transmission_Other=0
			transmission_Semi_Auto=1
		else:
			transmission_Manual=0
			transmission_Other=0
			transmission_Semi_Auto=0

		fueltype=float(request.form['fueltype'])
		if fueltype==1:
			fuelType_Electric=1
			fuelType_Hybrid=0
			fuelType_Other=0
			fuelType_Petrol=0
		if fueltype==2:
			fuelType_Electric = 0

			fuelType_Hybrid = 0
			fuelType_Other = 0
			fuelType_Petrol = 1
		if fueltype==3:
			fuelType_Electric=0
			fuelType_Hybrid=1
			fuelType_Other=0
			fuelType_Petrol=0
		if fueltype==5:
			fuelType_Electric=0
			fuelType_Hybrid=0
			fuelType_Other=1
			fuelType_Petrol=0
		else:
			fuelType_Electric=0
			fuelType_Hybrid=0
			fuelType_Other=0
			fuelType_Petrol=0
		final_arr=np.array([[year,mileage,engine_size,transmission_Manual,transmission_Other,transmission_Semi_Auto, fuelType_Electric, fuelType_Hybrid,fuelType_Other,fuelType_Petrol,brand]])

		pred=model.predict(final_arr)[0]
		if pred<0:
			return render_template('index.html', prediction_text="Sorry you cannot sell this car")
		else:
			return render_template('index.html', prediction_text="You Can Sell The Car at ${} ".format(pred))
		#output = round(pred[0], 2)
	else:


		return render_template('index.html')

if __name__=='__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)




