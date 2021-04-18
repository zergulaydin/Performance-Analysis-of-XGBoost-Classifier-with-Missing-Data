import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_excel("xgboost-results.xlsx",index_col=0)
print (df)
ax = df.plot.barh(rot=0)
plt.xlabel("F-score")
plt.legend(loc=4, prop={'size': 9},bbox_to_anchor=(0.90,1),ncol=len(df.columns))
plt.xticks(np.arange(0.0, 1.1, 0.1))
plt.show()