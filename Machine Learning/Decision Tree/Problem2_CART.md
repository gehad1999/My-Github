```python
import numpy as np
import pandas as pd
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#le.fit([1, 2, 2, 6])
# data = data.apply(le.fit_transform)

data2 = pd.read_csv('cardio_train.csv',sep=";")
data2.age=le.fit_transform(data2.cholesterol)
data2.height=le.fit_transform(data2.cholesterol)
data2.weight=le.fit_transform(data2.cholesterol)
data2.ap_hi=le.fit_transform(data2.cholesterol)
data2.ap_lo=le.fit_transform(data2.cholesterol)
print(data2)
print(len(data2),type(data2))
print("MAX for age",data2.loc[:,"age"].max(),"MIN for age",data2.loc[:,"age"].min())
print("MAX for height",data2.loc[:,"height"].max(),"MIN for height",data2.loc[:,"height"].min())
print("MAX for weight",data2.loc[:,"weight"].max(),"MIN for weight",data2.loc[:,"weight"].min())
print("MAX for ap_hi",data2.loc[:,"ap_hi"].max(),"MIN for ap_hi",data2.loc[:,"ap_hi"].min())
print("MAX for ap_lo",data2.loc[:,"ap_lo"].max(),"MIN for ap_lo",data2.loc[:,"ap_lo"].min())
#print((data2.age))
# classes,class_counts = np.unique(data2["age"],return_counts = True)
# print(classes,class_counts)
#features.gluc = features.gluc.map({1: 0, 2: 1, 3:2})
# features.gender = features.gender.map({1: 0, 2: 1})
# features.gluc = features.gluc.map({1: 0, 2: 1, 3:2})
# features.cholesterol = features.cholesterol.map({1: 0, 2: 1, 3:2})
train=data2.values[:63000,1:]  # train data_set
test=data2.values[63001:70000,1:]

#data1=data2.drop(['id'], axis=1)
#arr = data1.to_numpy()
#print(type(data1))
print(type(train))
#print(arr)
```

              id  age  gender  height  weight  ap_hi  ap_lo  cholesterol  gluc  \
    0          0    0       2       0       0      0      0            1     1   
    1          1    2       1       2       2      2      2            3     1   
    2          2    2       1       2       2      2      2            3     1   
    3          3    0       2       0       0      0      0            1     1   
    4          4    0       1       0       0      0      0            1     1   
    ...      ...  ...     ...     ...     ...    ...    ...          ...   ...   
    69995  99993    0       2       0       0      0      0            1     1   
    69996  99995    1       1       1       1      1      1            2     2   
    69997  99996    2       2       2       2      2      2            3     1   
    69998  99998    0       1       0       0      0      0            1     2   
    69999  99999    1       1       1       1      1      1            2     1   
    
           smoke  alco  active  cardio  
    0          0     0       1       0  
    1          0     0       1       1  
    2          0     0       0       1  
    3          0     0       1       1  
    4          0     0       0       0  
    ...      ...   ...     ...     ...  
    69995      1     0       1       0  
    69996      0     0       1       1  
    69997      0     1       0       1  
    69998      0     0       0       1  
    69999      0     0       1       0  
    
    [70000 rows x 13 columns]
    70000 <class 'pandas.core.frame.DataFrame'>
    MAX for age 2 MIN for age 0
    MAX for height 2 MIN for height 0
    MAX for weight 2 MIN for weight 0
    MAX for ap_hi 2 MIN for ap_hi 0
    MAX for ap_lo 2 MIN for ap_lo 0
    <class 'numpy.ndarray'>
    


```python

def class_counts(dataset):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in dataset:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts
print("\n Train Data \n",class_counts(train),"<-- classes label's counts ","\n Test Data \n",class_counts(test),"<-- classes label's counts ")
def gini(dataset):
    counts = class_counts(dataset)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(dataset))
        impurity -= prob_of_lbl**2
    return impurity
print (gini(train))


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)



def split_data(index,feature, data):
    left = np.where(data[:,feature] <= index)
    right = np.where(data[:,feature] > index)
    left_dataset = data[left]
    right_dataset = data[right]
    return left_dataset, right_dataset
current_uncertainty = gini(train)
left_dataset, right_dataset=split_data(2,4, train)
print("left_dataset \n",left_dataset,"\n","right_dataset \n", right_dataset)
gain= info_gain(left_dataset, right_dataset, current_uncertainty)
print("gaaaaaaain",gain)
def find_split(dataset):
    #initialize values
    best_feature, best_value, best_gain = 0, 0.0, 0.0
    current_uncertainty = gini(dataset)
    for feature in range(11):
        #get all unique classes for a feature
        unique_set = np.unique(dataset[:,feature])
        #print(unique_set)
        for index in unique_set:
            left_dataset, right_dataset = split_data(index,feature,dataset)
            #calculate the information gain
            gain= info_gain(left_dataset, right_dataset, current_uncertainty)
            #gain = get_info_gain(dataset, left_dataset, right_dataset)
            if len(left_dataset)==0 or len(right_dataset)==0:
               continue
            if gain > best_gain:
                best_feature = feature +1
                best_value = index
                best_gain = gain
    return best_feature, best_value


best_feature, best_value = find_split(train)
print("best_feature",best_feature,  "best_value",best_value)
#Decision tree algorithm
def decision_tree_learning(dataset,depth):
    #print((np.unique(dataset[:, 11])))
    if len(np.unique(dataset[:, 11])) == 1:
        #print(np.unique(dataset[:, 11]))
        attribute = np.unique(dataset[:,11])
        terminal_node = {'attribute':int(attribute),'leaf':1,'value':0,'left': None,'right': None}
        return terminal_node, depth
    else:
        attribute, value = find_split(dataset)
        l =np.where(dataset[:, attribute-1] <= value)
        r =np.where(dataset[:, attribute-1] > value)
        l_dataset = dataset[l]
        r_dataset = dataset[r]
        l_branch, l_depth = decision_tree_learning(l_dataset,depth+1)
        r_branch, r_depth = decision_tree_learning(r_dataset,depth+1)
        node = {'attribute': attribute,'leaf': 0, 'value': value, 'left': l_branch, 'right': r_branch}
        return node, max(l_depth,r_depth)
    
node,max_depth=decision_tree_learning(train,0)
print(node,max_depth)
```

    
     Train Data 
     {0: 31543, 1: 31457} <-- classes label's counts  
     Test Data 
     {0: 3477, 1: 3522} <-- classes label's counts 
    0.49999906827916346
    left_dataset 
     [[0 2 0 ... 0 1 0]
     [2 1 2 ... 0 1 1]
     [2 1 2 ... 0 0 1]
     ...
     [1 1 1 ... 0 1 1]
     [1 1 1 ... 0 1 0]
     [1 1 1 ... 0 1 1]] 
     right_dataset 
     []
    gaaaaaaain 0.0
    best_feature 1 best_value 0
    {'attribute': 1, 'leaf': 0, 'value': 0, 'left': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}}}}, 'right': {'attribute': 1, 'leaf': 0, 'value': 1, 'left': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}, 'right': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 2, 'left': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}, 'right': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 8, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}}}, 'right': {'attribute': 10, 'leaf': 0, 'value': 0, 'left': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}, 'right': {'attribute': 2, 'leaf': 0, 'value': 1, 'left': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 9, 'leaf': 0, 'value': 0, 'left': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}, 'right': {'attribute': 11, 'leaf': 0, 'value': 0, 'left': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}, 'right': {'attribute': 0, 'leaf': 0, 'value': 0.0, 'left': {'attribute': 0, 'leaf': 1, 'value': 0, 'left': None, 'right': None}, 'right': {'attribute': 1, 'leaf': 1, 'value': 0, 'left': None, 'right': None}}}}}}}}} 9
    


```python

def classifySample(sample, decisionTree):
    if not isinstance(decisionTree, dict):
        return decisionTree
    question = list(decisionTree.keys())[0]
    attribute, value = question.split(" <= ")
    if sample[attribute] <= float(value):
        answer = decisionTree[question][0]
    else:
        answer = decisionTree[question][1]
    return classifySample(sample, answer)

def decisionTreePredictions(dataFrame, decisionTree):
    dataFrame = pd.DataFrame(data=dataFrame,columns=["age","gender","height","weight","ap_hi","ap_lo"
        ,"cholesterol","gluc","smoke","alco","active","cardio"])
    predictions = dataFrame.apply(classifySample, axis = 1, args = (decisionTree,))
    return predictions

def calculateAccuracy(predictedResults, category):
    resultCorrect = predictedResults == category
    return resultCorrect.mean()

decisionTree = decision_tree_learning(train, 0)
#buildingTime = time.time() - startTime
decisionTreeTestResults = decisionTreePredictions(test, decisionTree)
accuracyTest = calculateAccuracy(decisionTreeTestResults, test[:, -1]) * 100
decisionTreeTrainResults = decisionTreePredictions(train, decisionTree)
accuracyTrain = calculateAccuracy(decisionTreeTrainResults, train[:, -1]) * 100
print(accuracyTest,accuracyTrain)

```

    0.0 0.0
    


```python
# returns what the model predicts given the data
def predict(data, model):
    tree = model
    while True:
        if tree['left'] is None and tree['right'] is None:
            return tree['leaf']
        attribute = tree['attribute']
        value = tree['value']

        if data[attribute] > value:
            tree = tree['right']
        else:
            tree = tree['left']

    return -1

def get_confusion_matrix(actual_labels, predicted_labels):
    #initialize confusion matrix
    cmat = np.zeros((4,4))
    #loops through data and counts number of actual-prediction occurences to create confusion matrix
    for i in range(len(predicted_labels)):
        cmat[actual_labels[i] -1, predicted_labels[i] -1] += 1

    return cmat

def classification_rate(confusion_matrix):
    classification = np.zeros((4))
    #the classification rate is the diagonal of the confusion matrix (TP+TN) over total
    for i in range(4):
        fn = int(confusion_matrix.sum(axis = 1)[i] - confusion_matrix[i,i])
        fp = int(sum(confusion_matrix[:,i]) - confusion_matrix[i,i])
        tp_tn = int(confusion_matrix.diagonal().sum(axis = 0))
        total = fn + fp + tp_tn
        classification[i] = tp_tn / total

    return classification
# Evaluate accuracy(classification rate) of the input model.
def evaluate(tree_model, test_data):
    actual_labels=[]
    predicted_labels = []

    for data in test_data:
        label = int(data[-1])
        predicted = predict(data, tree_model)
        actual_labels.append(label)
        predicted_labels.append(predicted)

    cmat = get_confusion_matrix(actual_labels,predicted_labels)
    class_rate = classification_rate(cmat)
    avg_class_rate = sum(class_rate) / len(class_rate)
    return avg_class_rate



dt, depth = decision_tree_learning(test,0)
print(evaluate(dt,test))


```

    0.7516073724817831
    


```python

```
