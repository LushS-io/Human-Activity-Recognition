# %% 
import pandas as pd
import numpy as np

# %%
labelstxt = pd.read_csv(filepath_or_buffer="./HAPT Data Set/activity_labels.txt", names=['label'])

labels = pd.DataFrame(columns=['labels', 'numbers'])
labels['labels'] = labelstxt['label'].apply(lambda x: [str(x).strip(' 123456789')])
labels['numbers'] = labelstxt['label'].apply(lambda x: [str(x).strip('ABCDEFGHIJKLMNOPQRSTUVWXYZ_ ')])
print(labels)

# %%
df_features_info = pd.read_csv(filepath_or_buffer='./HAPT Data Set/features.txt')



#%%
