import pandas as pd
from sklearn import preprocessing
import numpy as np
temp=pd.read_csv('dataset_58_vowel.csv')
temp.drop('Train_or_Test',axis=1,inplace=True)
le = preprocessing.LabelEncoder()
temp['Sex'] = le.fit_transform(temp.Sex.values)
temp['Speaker_Number'] = le.fit_transform(temp.Speaker_Number.values)
print(len(temp['Class'].unique())) # 11
dummy=pd.get_dummies(temp['Class'])
temp.drop('Class',axis=1,inplace=True)
temp=pd.concat([temp,dummy],axis=1)
print(type(temp))
fmt = '%d', '%1.1f', '%1.9f', '%1.9f'
np.savetxt(r'vowel.txt', temp.values, delimiter =',',fmt="%6f")
temp.to_csv('file1.csv')
temp=pd.read_csv('file1.csv')

with open('file1.csv', 'r') as inp, open('myfile.txt', 'w') as out:
    for line in inp:
        line = line.replace(',', ',')
#temp.drop(',',axis=1,inplace=True)
# oh=preprocessing.OneHotEncoder()
# temp['Class'] = oh.fit_transform(temp.Class.values).toarray()
print(temp)