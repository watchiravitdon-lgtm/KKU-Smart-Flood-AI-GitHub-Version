import numpy as np
import pandas as pd
from pathlib import Path
rng=np.random.default_rng(42); n=1200
rainfall=np.clip(rng.gamma(2.2,25,n),0,250)
water=np.clip(rng.normal(.85,.38,n),.05,2.5)
humidity=np.clip(rng.normal(78,11,n),35,100)
temperature=np.clip(rng.normal(29,3.5,n),18,40)
score=rainfall*.45+water*38+humidity*.35-temperature*1.2+rng.normal(0,7,n)
risk=np.select([score<55,score<95],["LOW","MEDIUM"],default="HIGH")
df=pd.DataFrame({'Rainfall_mm':rainfall.round(2),'Water_Level_m':water.round(2),'Humidity_percent':humidity.round(2),'Temperature_C':temperature.round(2),'Risk':risk})
Path('data').mkdir(exist_ok=True); df.to_csv('data/flood_data.csv',index=False)
print(f'Created {len(df)} rows'); print(df.Risk.value_counts())
