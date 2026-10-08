from pathlib import Path
import joblib,pandas as pd
import matplotlib.pyplot as plt
def load_data(): return pd.read_csv('data/flood_data.csv')
def feature_importance():
 b=joblib.load('models/flood_model.pkl'); return pd.DataFrame({'Feature':b['features'],'Importance':b['model'].feature_importances_}).sort_values('Importance',ascending=False)
def create_feature_importance_chart():
 r=feature_importance(); Path('images').mkdir(exist_ok=True); plt.figure(figsize=(9,5)); plt.barh(r['Feature'][::-1],r['Importance'][::-1]); plt.xlabel('Importance'); plt.ylabel('Feature'); plt.title('AI Feature Importance'); plt.tight_layout(); plt.savefig('images/feature_importance.png',dpi=160); plt.close()
if __name__=='__main__': print(feature_importance()); create_feature_importance_chart()
