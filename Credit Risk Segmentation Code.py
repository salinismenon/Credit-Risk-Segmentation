#%% Libraries Used
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.vq import whiten, kmeans, vq


#%%Load Credit Dataset

os.chdir("D:\OneDrive - King's College London\Salini\Projects\Lending Club\Loan Data")
credit_data = pd.read_csv("accepted_2007_to_2018Q4.csv")

#%% First look at the data

print(credit_data.shape)
print(credit_data.columns)
credit_data.info()
credit_data.head()
print(credit_data.head(15))


#%% Understanding the available features

#columns = credit_data.columns.tolist()
#for i, col in enumerate(columns):
#    print(i,col)

#for i, col in enumerate(columns):
# if credit_data[col].dtype == 'object':
#        print(f"\nColumn: {col}")
#        print(col, credit_data[col].unique())

#%% Matching column names to descriptions

#def to_camel(s):
#   parts = s.split('_')
#  return parts[0] + ''.join(word.capitalize() for word in parts[1:])

#aligned = []

#for col in credit_data.columns:
#    camel = to_camel(col)
#    aligned.append((col, camel))

#for item in aligned:
#    print(item)

#%% Feature selection

credit_data["inc_to_loan_ratio"] = credit_data["annual_inc"]/credit_data["loan_amnt"]
credit_data["avg_fico"] = (credit_data["fico_range_high"] + credit_data["fico_range_low"])/2
credit_data_filtered = credit_data[["inc_to_loan_ratio", "dti", "avg_fico", "delinq_2yrs","delinq_amnt", "mths_since_last_delinq", "bc_util", "revol_util", "all_util", "total_bc_limit", "int_rate", "purpose", "emp_length", "inq_last_12m", "mths_since_recent_bc", "open_acc_6m", "home_ownership", "loan_status"]]
credit_data_filtered.head()
"""print(credit_data_filtered["loan_status"].unique())
credit_data_filtered = credit_data_filtered.replace([np.inf,-np.inf],np.nan)
credit_data_filtered = credit_data_filtered.dropna()"""

#%% EDA

credit_data_filtered.info()
credit_data_filtered.isna().sum()
credit_data_filtered.drop(columns=['mths_since_last_delinq','mths_since_recent_bc','bc_util'], inplace=True)
credit_data_filtered = credit_data_filtered.replace([np.inf,-np.inf],np.nan)
credit_data_filtered = credit_data_filtered.dropna()
credit_data_filtered.isna().sum()
scaled_data = whiten(credit_data_filtered[["inc_to_loan_ratio", "dti", "avg_fico", "delinq_2yrs"]])




#%% K means Clustering
num_clusters = range(3,10)
distortions=[]

for i in num_clusters:
    centroids, distortion = kmeans(scaled_data,i) #"avg_fico", "delinq_2yrs","delinq_amnt", "mths_since_last_delinq", "bc_util", "revol_util", "all_util", "total_bc_limit", "int_rate", "inq_last_12m", "mths_since_recent_bc", "open_acc_6m"]],i)
    distortions.append(distortion)
    
elbow_plot_data = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot_data)
plt.show()
print("done")
"""
centroids, distortion = kmeans(scaled_data[["inc_to_loan_ratio", "dti", "avg_fico", "delinq_2yrs"]],3)
scaled_data["cluster_labels"], distortion = vq(scaled_data[["inc_to_loan_ratio", "dti", "avg_fico", "delinq_2yrs"]], centroids)
print(scaled_data["cluster_labels"])"""

#sns.scatterplot(x="delinq_2yrs", y='delinq_amnt', data = credit_data_filtered, hue='cluster_labels')
#plt.show()
