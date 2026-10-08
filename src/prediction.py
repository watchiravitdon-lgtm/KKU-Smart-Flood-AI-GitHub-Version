import joblib,pandas as pd
def load_model(): return joblib.load('models/flood_model.pkl')
def predict_risk(rainfall,water_level,humidity,temperature):
 b=load_model(); m=b['model']; X=pd.DataFrame([{'Rainfall_mm':rainfall,'Water_Level_m':water_level,'Humidity_percent':humidity,'Temperature_C':temperature}]); p=m.predict(X)[0]; q=m.predict_proba(X)[0]; return p,{c:float(v) for c,v in zip(m.classes_,q)}
