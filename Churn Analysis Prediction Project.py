#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### Churn Data Analysis:

# In[3]:


File_Path = r"‪C:\Users\SHUBHAM PAWAR\Desktop\PowerBI\Credit Card Analysis Project\ML Part\Prediction_Data_Churn.xlsx"


# In[4]:


import os
import pandas as pd

# Correct file path
file_path = r"C:\Users\SHUBHAM PAWAR\Desktop\PowerBI\Credit Card Analysis Project\ML Part\Prediction_Data_Churn.xlsx"

# Check if the file exists
if os.path.exists(file_path):
    try:
        # Read the data from the specified sheet into a pandas dataframe
        sheet_name = 'vw_ChurnData'  # Change this to the correct sheet name if needed
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print("Data read successfully.")
        print(df.head())  # Print the first few rows of the dataframe
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("File does not exist at the specified path.")


# In[5]:


df.head()


# In[6]:


df.columns


# ### Data Preprocessing:

# In[7]:


#Drop Columns thar won't be used for prediction
df=df.drop(['Customer_ID','Churn_Category','Churn_Reason'],axis=1)
#List the columns to be label encoded:
columns_to_encode = ['Gender', 'Married', 'State','Value_Deal',
       'Phone_Service', 'Multiple_Lines', 'Internet_Service', 'Internet_Type',
       'Online_Security', 'Online_Backup', 'Device_Protection_Plan',
       'Premium_Support', 'Streaming_TV', 'Streaming_Movies',
       'Streaming_Music', 'Unlimited_Data', 'Contract', 'Paperless_Billing',
       'Payment_Method']
#Encode Categorical variables except target variables
label_encoders = {}
for column in columns_to_encode:
    label_encoders[column] = LabelEncoder()
    df[column]= label_encoders[column].fit_transform(df[column])

#manually encode the target variable 'Customer Status'. This column we have not included in above column beacuse
#with label encoder, aplhabatically Churned will be 0 and stayed will be 1, hence need manual mapping.

df['Customer_Status'] =df['Customer_Status'].map({'Stayed':0,'Churned':1})

#split the data into features and targets

X= df.drop('Customer_Status',axis=1)
y=df['Customer_Status']


# In[8]:


#Split the data into training and testing:
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
#training data = 80%, testing data =20%
#random_state = To keep check on randomness, in order to reproduce the same no. of results data


# In[9]:


# Train the Model
#initialize the random forest model:
rf_model =RandomForestClassifier(n_estimators=100,random_state=42)  #here n_estimators is nothing but the no.of d.trees
#Train the model:
rf_model.fit(X_train,y_train)
#let's do predictions:

y_pred = rf_model.predict(X_test)

#evaluate the model:
print('Confusion matrix:')
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))


# In[10]:


# 86% predictions for class 0 (i.e output) are true (accuracy of positive predictions made by model)
#78% predictions for class 1 (Recall measures ability to identify all the positive instances)


# In[11]:


#feature scaling using feature importance:
importances=rf_model.feature_importances_
indices = np.argsort(importances)[::-1]  #[::-1] is for highest to lowest

#Plot the feature importances:

plt.figure(figsize=(15,6))
sns.barplot(x=importances[indices],y=X.columns[indices])
plt.title('Feature Importance')
plt.xlabel('Relative Importance')
plt.ylabel('Feature Name')
plt.show()


# ### Join Data Analysis sheet:

# In[12]:


File_Path = r"‪C:\Users\SHUBHAM PAWAR\Desktop\PowerBI\Credit Card Analysis Project\ML Part\Prediction_Data_Join.xlsx"


# In[13]:


import os
import pandas as pd

# Correct file path
file_path = r"C:\Users\SHUBHAM PAWAR\Desktop\PowerBI\Credit Card Analysis Project\ML Part\Prediction_Data_Join.xlsx"

# Check if the file exists
if os.path.exists(file_path):
    try:
        # Read the data from the specified sheet into a pandas dataframe
        sheet_name = 'vw_JoinData'  # Change this to the correct sheet name if needed
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print("Data read successfully.")
        print(df.head())  # Print the first few rows of the dataframe
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("File does not exist at the specified path.")


# In[14]:


# Define the sheet name to read data from
sheet_name = 'vw_JoinData'

# Read the data from the specified sheet into a pandas DataFrame
df1 = pd.read_excel(file_path, sheet_name=sheet_name)

# Display the first few rows of the fetched data
print(df1.head())

# Retain the original DataFrame to preserve unencoded columns
original_data = df1.copy()

# Retain the Customer_ID column
customer_ids = df1['Customer_ID']


# In[15]:


# Drop columns that won't be used for prediction in the encoded DataFrame
df1 = df1.drop(['Customer_ID', 'Customer_Status', 'Churn_Category', 'Churn_Reason'], axis=1)
# Encode categorical variables using the saved label encoders
for column in df1.select_dtypes(include=['object']).columns:
    df1[column] = label_encoders[column].transform(df1[column])


# In[16]:


# Make predictions
new_predictions = rf_model.predict(df1)


# In[17]:


# Add predictions to the original DataFrame
original_data['Customer_Status_Predicted'] = new_predictions


# In[18]:


# Filter the DataFrame to include only records predicted as "Churned"
original_data = original_data[original_data['Customer_Status_Predicted'] == 1]


# In[19]:


# Save the results
original_data.to_csv(r"C:\Users\SHUBHAM PAWAR\Desktop\PowerBI\Credit Card Analysis Project\ML Part.Predictions.csv", index=False)


# In[ ]:




