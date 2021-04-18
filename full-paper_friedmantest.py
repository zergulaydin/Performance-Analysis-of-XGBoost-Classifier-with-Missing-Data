from scipy.stats import wilcoxon, friedmanchisquare, rankdata, wilcoxon
import numpy as np
import pandas as pd
import scikit_posthocs as sp


df=pd.read_excel("xgboost-results.xlsx",sheet_name='fscore')
print(df)


Model_names = df.drop('Dataset', axis=1).columns
fscores= df[Model_names].values

ranks = np.array([rankdata(-p) for p in fscores])


data1=df.iloc[:,1]
data2=df.iloc[:,2]
data3=df.iloc[:,3]
data4=df.iloc[:,4]
data5=df.iloc[:,5]
data6=df.iloc[:,6]


print(friedmanchisquare(data1,data2,data3,data4))





"""
print(sp.posthoc_nemenyi_friedman(data))
print(wilcoxon(data1, data6, zero_method='zsplit'))

average_ranks = np.mean(ranks, axis=0)
print('\n'.join('{} average rank: {}'.format(a, r) for a, r in zip(Model_names, average_ranks)))

print(sp.posthoc_nemenyi_friedman(fscores))
print(sp.posthoc_wilcoxon(fscores))
x = [[1,2,3,4,5], [35,31,75,40,21], [10,6,9,6,1]]
print(x)
print(sp.posthoc_wilcoxon(x))"""