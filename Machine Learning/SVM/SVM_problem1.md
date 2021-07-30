```python
# Importing the necessary libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statistics
```


```python
df = pd.read_csv('data.csv')
#Separating features and labels
features=df.iloc[:, :-1]
label=df.iloc[:,-1]
```


```python
svc = SVC(kernel='linear')
accu=[]
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.4, random_state=1+i)
    # Default Linear kernel
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    acc = metrics.accuracy_score(y_test,y_pred)
    accu.append(acc)
```


```python
print(accu)
```

    [0.79375, 0.709375, 0.75, 0.7625, 0.753125, 0.725, 0.746875, 0.7375, 0.721875, 0.740625]
    


```python
mea=statistics.mean(accu)
print("mean = ",mea)
```

    mean =  0.7440625
    


```python
# After scaling the data
z=(features.min()-features.max())
X=(features-features.mean(axis=0))/z
print("Scaled \n",X)
```

    Scaled 
               1st       2nd       3rd       4th       5th       6th       7th  \
    0    0.527500  0.215460 -0.363125  0.126807 -0.723437 -0.400625 -0.108750   
    1    0.194167 -0.402187  0.136875 -0.178925  0.276563  0.099375  0.224583   
    2   -0.472500  0.127224 -0.363125  0.069482  0.276563 -0.150625 -0.108750   
    3    0.527500 -0.313952  0.136875 -0.299944  0.276563 -0.150625 -0.108750   
    4    0.527500 -0.049246 -0.113125 -0.108861  0.276563  0.099375 -0.108750   
    ..        ...       ...       ...       ...       ...       ...       ...   
    795 -0.472500  0.171342  0.136875  0.056744  0.026563  0.349375  0.224583   
    796  0.527500  0.038989  0.136875 -0.274467 -0.723437 -0.400625 -0.108750   
    797 -0.472500  0.127224 -0.363125  0.120438  0.276563  0.349375  0.224583   
    798 -0.472500 -0.049246 -0.113125  0.158654 -0.723437 -0.400625 -0.442083   
    799  0.194167  0.171342  0.136875  0.107699 -0.723437  0.349375 -0.108750   
    
              8th       9th      10th  ...     15th  16th     17th  18th    19th  \
    0   -0.386250  0.455417 -0.564174  ...  0.03375  0.23  0.10125 -0.09  0.0375   
    1    0.280417  0.455417  0.239397  ...  0.03375  0.23  0.10125 -0.09  0.0375   
    2   -0.052917  0.455417 -0.242746  ...  0.03375  0.23  0.10125 -0.09  0.0375   
    3   -0.386250  0.122083 -0.171317  ...  0.03375  0.23  0.10125  0.91  0.0375   
    4   -0.386250 -0.544583 -0.314174  ...  0.03375 -0.77  0.10125 -0.09  0.0375   
    ..        ...       ...       ...  ...      ...   ...      ...   ...     ...   
    795 -0.386250  0.122083  0.239397  ...  0.03375  0.23  0.10125 -0.09  0.0375   
    796 -0.386250  0.122083 -0.278460  ...  0.03375  0.23 -0.89875 -0.09  0.0375   
    797 -0.386250  0.122083  0.239397  ...  0.03375  0.23  0.10125 -0.09  0.0375   
    798 -0.386250 -0.211250 -0.332031  ...  0.03375 -0.77  0.10125 -0.09  0.0375   
    799  0.280417  0.455417  0.007254  ...  0.03375 -0.77  0.10125 -0.09  0.0375   
    
          20th    21th     22th     23th     24th  
    0    0.175 -0.2875  0.02125  0.20125 -0.37125  
    1    0.175 -0.2875  0.02125  0.20125 -0.37125  
    2    0.175 -0.2875  0.02125 -0.79875  0.62875  
    3    0.175  0.7125  0.02125  0.20125 -0.37125  
    4    0.175  0.7125  0.02125  0.20125 -0.37125  
    ..     ...     ...      ...      ...      ...  
    795 -0.825  0.7125  0.02125  0.20125 -0.37125  
    796  0.175  0.7125  0.02125  0.20125 -0.37125  
    797 -0.825  0.7125  0.02125 -0.79875  0.62875  
    798  0.175 -0.2875  0.02125  0.20125 -0.37125  
    799  0.175 -0.2875 -0.97875  0.20125  0.62875  
    
    [800 rows x 24 columns]
    


```python
accu2=[]
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.4, random_state=1+i)
    # Default Linear kernel
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    acc = metrics.accuracy_score(y_test,y_pred)
    accu2.append(acc)
```


```python
print(accu2)
```

    [0.784375, 0.71875, 0.753125, 0.759375, 0.728125, 0.734375, 0.75, 0.734375, 0.7375, 0.721875]
    


```python
mea2=statistics.mean(accu2)
print("mean = ",mea2)
```

    mean =  0.7421875
    


```python
# The difference between the data before scaling and after scaling doesn't make a big difference on the accuracy.
# (the range of features values not wide) Features not have large difference on the range of their values.
```


```python

```
