
# Read Data
```python
import pandas as pd
import numpy as np
# Load the data
df = pd.read_excel('A-data.xlsx', names=['Early registration ',' Finished homework II', 'Senior','Likes Coffee','Liked The Last homework','A'])
print("Data \n",df)
print("\n",type(df))
```

    Data 
         Early registration    Finished homework II  Senior  Likes Coffee  \
    0                     1                      1       0             0   
    1                     1                      1       1             0   
    2                     0                      0       1             0   
    3                     0                      1       1             0   
    4                     0                      1       1             0   
    5                     0                      0       1             1   
    6                     1                      0       0             0   
    7                     0                      1       0             1   
    8                     0                      0       1             0   
    9                     1                      0       0             0   
    10                    1                      1       1             0   
    11                    0                      1       1             1   
    12                    0                      0       0             0   
    13                    1                      0       0             1   
    
        Liked The Last homework  A  
    0                         1  1  
    1                         1  1  
    2                         0  0  
    3                         1  0  
    4                         0  1  
    5                         1  1  
    6                         1  0  
    7                         1  1  
    8                         1  1  
    9                         0  0  
    10                        0  1  
    11                        1  0  
    12                        1  0  
    13                        0  1  
    
     <class 'pandas.core.frame.DataFrame'>
    
# Entropy function
# Information Gain function

```python

```

```python
def calculate_entropy(target_col):
    classes,class_counts = np.unique(target_col,return_counts = True)
    entropy_value = np.sum([(-class_counts[i]/np.sum(class_counts))*np.log2(class_counts[i]/np.sum(class_counts))
                            for i in range(len(classes))])
    return entropy_value

print("\n Entropy(A)=",calculate_entropy(df["A"]))
def calculate_information_gain(dataset,feature,label="class"):
    # calculate the dataset entropy
    dataset_entropy = calculate_entropy(dataset[label])
    values ,feat_counts= np.unique( dataset[feature] , return_counts=True)
    weighted_feature_entropy= np.sum([(feat_counts[i]/np.sum(feat_counts))*calculate_entropy(dataset.where(dataset[feature]==values[i]).
                                     dropna()[label]) for i in range (len(values))])
    #formula for information gain
    feature_info_gain = dataset_entropy - weighted_feature_entropy
    return feature_info_gain
print("\n Information Gain for feature Finished homework II for total dataset =",calculate_information_gain(df,' Finished homework II','A'))
```

    
     Entropy(A)= 0.9852281360342515
    
     Information Gain for feature Finished homework II for total dataset = 0.06105378373381021
    
 
 # Decision Tree 1st Depth (Root Node) 

```python
def DT1():
    item_values = [calculate_information_gain(df, feature, "A") for feature in
                   df.columns[:-1]]  # Return the infgain values
    best_feature_index = np.argmax(item_values)
    best_feature = df.columns[:-1][best_feature_index]
    features = [i for i in df.columns[:-1] if i != best_feature]
    sub_data = df.where(df[best_feature] == 0).dropna()
    sub_data3 = df.where(df[best_feature] == 1).dropna()
    classes1st, class_counts1st = np.unique(df[best_feature], return_counts=True)
    classes1sst, class_counts1sst = np.unique(df["A"], return_counts=True)  # number of +ve and -ve values
    print("ROOT Node is",best_feature)
    print("Entropy = ",calculate_entropy(df[best_feature]))
    print(classes1st, class_counts1st, "<-- BF classes counts")
    print(classes1sst, class_counts1sst,"<-- Label classes counts")  # number of +ve and -ve values
    
    
DT1()
```

    ROOT Node is  Finished homework II
    Entropy =  1.0
    [0 1] [7 7] <-- BF classes counts
    [0 1] [6 8] <-- Label classes counts
    
# Decision Tree 2nd Depth

```python

# parent root
def DT2():
    item_values = [calculate_information_gain(df, feature, "A") for feature in
                   df.columns[:-1]]  # Return the infgain values
    best_feature_index = np.argmax(item_values)
    best_feature = df.columns[:-1][best_feature_index]
    features = [i for i in df.columns[:-1] if i != best_feature]
    sub_data = df.where(df[best_feature] == 0).dropna()
    sub_data3 = df.where(df[best_feature] == 1).dropna()
    classes1st, class_counts1st = np.unique(df[best_feature], return_counts=True)
    classes1sst, class_counts1sst = np.unique(df["A"], return_counts=True)  # number of +ve and -ve values
    print("ROOT Node is",best_feature)
    print("Entropy = ",calculate_entropy(df[best_feature]))
    print(classes1st, class_counts1st, "<-- BF classes counts")
    print(classes1sst, class_counts1sst,"<-- Label classes counts")  # number of +ve and -ve values

    item_values2 = [calculate_information_gain(sub_data, feature, "A") for feature in
                    features]  # Return the infgain values
    best_feature_index2 = np.argmax(item_values2)
    best_feature2 = features[best_feature_index2]
    features2 = [i for i in features if i != best_feature2]
    sub_data2 = sub_data.where(sub_data[best_feature2] == 0).dropna()
    classes2nd, class_counts2nd = np.unique(sub_data[best_feature2], return_counts=True)
    classes2nnd, class_counts2nnd = np.unique(sub_data["A"], return_counts=True)  # number of +ve and -ve values
    print("Child Node is",best_feature2, "when Root class == 0")
    print("Entropy = ",calculate_entropy(sub_data[best_feature2]))
    #print(best_feature2, features2)
    print(classes2nd, class_counts2nd,"<-- BF classes counts")
    print(classes2nnd, class_counts2nnd,"<-- Label classes counts")  # number of +ve and -ve values
    

    item_values3 = [calculate_information_gain(sub_data3, feature, "A") for feature in
                    features]  # Return the infgain values
    best_feature_index3 = np.argmax(item_values3)
    best_feature3 = features[best_feature_index3]
    features3 = [i for i in features if i != best_feature3]
    sub_data4 = sub_data3.where(sub_data3[best_feature3] == 0).dropna()
    classes, class_counts = np.unique(sub_data3[best_feature3], return_counts=True)
    classes2, class_counts2 = np.unique(sub_data3["A"], return_counts=True)  # number of +ve and -ve values
    print("Child Node is",best_feature3, "when Root class == 1")
    print("Entropy = ",calculate_entropy(sub_data3[best_feature3]))  # show entropy at this node
    print(classes, class_counts,"<-- BF classes counts")
    print(classes2, class_counts2,"<-- Label classes counts")  # number of +ve and -ve values
   

DT2()
```

    ROOT Node is  Finished homework II
    Entropy =  1.0
    [0 1] [7 7] <-- BF classes counts
    [0 1] [6 8] <-- Label classes counts
    Child Node is Likes Coffee when Root class == 0
    Entropy =  0.863120568566631
    [0. 1.] [5 2] <-- BF classes counts
    [0. 1.] [4 3] <-- Label classes counts
    Child Node is Early registration  when Root class == 1
    Entropy =  0.9852281360342515
    [0. 1.] [4 3] <-- BF classes counts
    [0. 1.] [2 5] <-- Label classes counts
    


```python

```
