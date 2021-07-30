#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
# Load the data
df = pd.read_excel('A-data.xlsx', names=['Early registration ',' Finished homework II', 'Senior','Likes Coffee','Liked The Last homework','A'])
print("Data \n",df)
print("\n",type(df))


# In[ ]:





# In[13]:


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


# In[39]:


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


# In[38]:


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
    
    
#     print("Best Feature classes -->",classes2nd,"  Counts For each class -->", class_counts2nd)
#     print("Number of +ve and -ve values for A at the Best Feature \nclasses --> ",classes2nnd,
#           "  Counts For each class -->", class_counts2nnd)  # number of +ve and -ve values
#     print("Best feature (Root) -->",best_feature,"\nRest features are", features)
#     print("entropy(dataset[best feature])-->",calculate_entropy(sub_data[best_feature2]))
   

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


# In[ ]:




