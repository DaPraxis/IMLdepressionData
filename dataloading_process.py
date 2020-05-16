import pandas as pd
import streamlit as st
import os
import urllib

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/DaPraxis/IMLdepressionData/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

@st.cache
def load_metadata(url):
    return pd.read_csv(url)

# This function uses some Pandas magic to summarize the metadata Dataframe.
@st.cache
def create_summary(metadata):
    df = metadata[['FHAMD1', 'FHAMD2', 'FHAMD3', 'FHAMD4', 'FHAMD5','FHAMD6', 'FHAMD7', 'FHAMD8', 'FHAMD9', 'FHAMD10', 
        'FHAMD11', 'FHAMD12', 'FHAMD13', 'FHAMD14', 'FHAMD15', 'FHAMD16', 'FHAMD17','LHAMD1', 'LHAMD2', 'LHAMD3', 
        'LHAMD4', 'LHAMD5', 'LHAMD6', 'LHAMD7', 'LHAMD8', 'LHAMD9', 'LHAMD10', 'LHAMD11', 'LHAMD12', 'LHAMD13',
        'LHAMD14', 'LHAMD15', 'LHAMD16', 'LHAMD17', 'DRUG1', 'DRUG2', 'DRUG']]
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

    for i in range(len(df)):
        label = df.loc[i,'DRUG']
        if label == 'DuloxetineDuloxetine':
            df.loc[i,'DRUG'] = 'Duloxetine'
        if label == 'VenlafaxVenlafax':
            df.loc[i,'DRUG'] = 'Venlafax'
    df = df[(df.DRUG!='VenlafaxPlacebo')&(df.DRUG!='DuloxetinePlacebo')&(df.DRUG!='EscitalopramPlacebo')].reset_index()
    df_first = df[df.columns[df.columns.str.contains('FHAMD')]]
    df_last = df[df.columns[df.columns.str.contains('LHAMD')]]
    diff = df_last.values - df_first.values

    col_names = []
    for i in range(17):
        name = 'DHAMD' + str(i+1)
        col_names.append(name)

    #create dataframe
    finaldf = pd.DataFrame(diff,columns=col_names)

    #add label column
    finaldf['DRUG'] = df['DRUG']
    return finaldf


def file_selector(folder_path='./dataset/'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file to load data', filenames)
    return os.path.join(folder_path, selected_filename)