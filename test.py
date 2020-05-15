import streamlit as st
import numpy as np
import pandas as pd
import missingno as msno
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px

st.title('START Clinic Data Analysis')
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file to load data', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)

df = pd.read_csv(filename)

# show missing values density

st.write('Missing data matrix')

nans = pd.concat([df.isnull().sum(), (df.isnull().sum() / df.shape[0]) * 100], axis=1, keys=['Num_NaN', 'NaN_Percent'])
nans = nans[nans.Num_NaN > 0]
nans = nans.sort_values(by=['NaN_Percent'], ascending=False)
cols_with_nans = nans.index.tolist()
msno.matrix(df=df[cols_with_nans[:25]], figsize=(30, 15), color=(0.24, 0.77, 0.77))
# st.pyplot()

# preprocess data

df.dropna(how='all',inplace=True)

for i in range(len(df)):
    
    label = df.loc[i,'DRUG']
    
    if label == 'DuloxetineDuloxetine':
        
        df.loc[i,'DRUG'] = 'Duloxetine'
        
    if label == 'VenlafaxVenlafax':
        
        df.loc[i,'DRUG'] = 'Venlafax'

notreat_idx = df[(df['DRUG1'].isnull()) & (df['DRUG2'].isnull())].index
df.drop(notreat_idx,inplace=True)

ctreat_idx =  df[(df['DRUG1'].notnull()) & (df['DRUG2'].notnull())][df['DRUG1']!=df['DRUG2']].index
df.drop(ctreat_idx,inplace=True)

#drop columns drug 1 and drug 2
df.drop(['DRUG1','DRUG2'], axis=1, inplace=True)

#drop rows with NaNs
df.dropna(inplace=True)

#reset indices and show dataframe
df = df.reset_index(drop=True)

df_first = df[df.columns[df.columns.str.contains('FHAMD')]]

df_last = df[df.columns[df.columns.str.contains('LHAMD')]]

