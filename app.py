from flask import Flask,request,render_template
import pandas as pd
import pickle
import numpy as np


app=Flask(__name__)

model1=pickle.load(open("kidney.pkl",'rb'))
model2=pickle.load(open("CKD.pkl",'rb'))
model3=pickle.load(open("diabetes.pkl",'rb'))

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/home',methods=['GET','POST]'])
def home():
    return render_template("home.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    return render_template("predict.html")

@app.route('/kidney',methods=['GET','POST'])
def kidney():
    return render_template("kidney.html")

@app.route('/lung',methods=['GET','POST'])
def lung():
    return render_template("lung.html")

@app.route('/diabetes',methods=['GET','POST'])
def diabetes():
    return render_template("diabetes.html")

@app.route('/result1',methods=['POST'])
def result1():
    blood_urea=request.form.get("blood_urea")
    blood_glucose_random=request.form.get("blood_glucose_random")
    coronary=request.form.get("coronary_artery_disease")
    anemia=request.form.get("anemia")
    pus_cell=request.form.get("pus_cell")
    rbc=request.form.get("red_blood_cells")
    diabetesmellitus=request.form.get("diabetesmellitus")
    pedal_edema=request.form.get("pedal_edema")
    
    inputs=[blood_urea,blood_glucose_random,coronary,anemia,pus_cell,rbc,diabetesmellitus,pedal_edema]
    features=[np.array(inputs)]
    
    cols=['blood_urea','blood_glucose_random','coronary_artery_disease','anemia','pus_cell','red_blood_cells','diabetesmellitus',
                'pedal_edema']
    df=pd.DataFrame(features,columns=cols)
    output=model1.predict(df)
    a=[1]
    if output==a:
        return render_template("kidney.html",items="Oops You Have chronic kidney disease....!")
    else:
        return render_template("kidney.html",item="You are Normal.....!")

@app.route('/result2',methods=['POST'])
def result2():
    anexity=request.form.get("an")
    yellow_finger=request.form.get("yellow")
    peer_pressure=request.form.get("peer")
    Chronic=request.form.get("ch")
    wheezing=request.form.get("wh")
    cough=request.form.get("coug")
    SOB=request.form.get("sob")
    shallowing=request.form.get("shaw")
    chest_pain=request.form.get("cp")
    
    inputs=[anexity,yellow_finger,peer_pressure,Chronic,wheezing,cough,SOB,shallowing,chest_pain]
    features=[np.array(inputs)]
    
    
    column=['ANXIETY','YELLOW_FINGERS','PEER_PRESSURE','CHRONIC DISEASE','WHEEZING','COUGHING',
           'SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN']
    
    df=pd.DataFrame(features,columns=column)
    output=model2.predict(df)
    b=[1]
    
    if output==b:
        return render_template("lung.html",item="Oops you have lung Tumor......!")
    else:
        return render_template("lung.html",items="You are Normal.....!")

@app.route('/result3',methods=['POST'])
def result3():
    preg=request.form.get("preg")
    glucose=request.form.get("glucose")
    BP=request.form.get("BP")
    skin=request.form.get("Skin")
    insulin=request.form.get("insulin")
    bmi=request.form.get("bmi")
    diabetes=request.form.get("diabetes")
    Age=request.form.get("num")
    
    
    inputs=[preg,glucose,BP,skin,insulin,bmi,diabetes,Age]
    features=[np.array(inputs)]

    column=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


    df=pd.DataFrame(features,columns=column)

    output=model3.predict(df)
    
    a=[1]
    if output==a:
        return render_template("diabetes.html",items="Oops you have diabetes.....!")
    else:
        return render_template("diabetes.html",item="You are Normal.....!")
        
if __name__=="__main__":
    app.run(debug=True)