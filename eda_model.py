import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

os.chdir("C:\\Users\\ayush\\OneDrive\\Desktop\\Carpred\\Data")

import glob
extension = 'csv'
final_df=pd.DataFrame()
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
for f in all_filenames:
    a=pd.read_csv(f)
    a["Company"]=f[:-4]
    final_df=final_df.append(a)


final_df.drop(labels=["tax(£)","mpg","tax","model"],inplace=True,axis=1)
final_df=pd.get_dummies(final_df,columns=['transmission','fuelType'],drop_first=True)
company_dict=final_df.groupby(["Company"])["price"].mean().to_dict()
final_df["company_wrt_price"]=final_df["Company"].map(company_dict)
final_df.drop("Company",axis=1,inplace=True)


x=final_df.drop("price",axis=1)
y=final_df.loc[:,"price"]
x["year"]=2020-x["year"]
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.2)


rf=RandomForestRegressor()
rf=rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)


import pickle
# open a file, where you ant to store the data
filename = 'C:\\Users\\ayush\\OneDrive\\Desktop\\Carpred\\car_model.sav'
pickle.dump(rf, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)
