from pathlib import Path
import joblib,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
features=['Rainfall_mm','Water_Level_m','Humidity_percent','Temperature_C']
df=pd.read_csv('data/flood_data.csv'); X,y=df[features],df['Risk']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=150,max_depth=8,random_state=42,class_weight='balanced'); model.fit(Xtr,ytr)
pred=model.predict(Xte); acc=accuracy_score(yte,pred); cm=confusion_matrix(yte,pred,labels=['LOW','MEDIUM','HIGH'])
print(f'Accuracy: {acc:.4f}'); print(classification_report(yte,pred)); print(cm)
Path('models').mkdir(exist_ok=True); joblib.dump({'model':model,'features':features,'classes':list(model.classes_),'accuracy':acc,'confusion_matrix':cm.tolist()},'models/flood_model.pkl')
