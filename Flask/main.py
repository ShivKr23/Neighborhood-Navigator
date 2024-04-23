from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

data=pd.read_csv("Cleaned_Bangluru_Data.csv")

app=Flask(__name__)

pipe=pickle.load(open("RidgeModel.pkl","rb"))

@app.route('/')
def index():
    
    locations=sorted(data['location'].unique())
    return render_template('index.html',locations = locations)

@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get( 'location' )
    bhk=request.form.get( 'bhk' )
    bath=request.form.get( 'bath' )
    sqft=request.form.get('total_sqft')
    
   
    
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    
    
    try:
        prediction = pipe.predict(input_data)[0] * 1e5
        # Render prediction in a user-friendly format
        return "₹"+str(np.round(prediction, 2))
    except Exception as e:
        return "Something went wrong try again!!"
    
# if __name__=='__main__':
#     app.run(debug=True,port=5001)
