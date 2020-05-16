import streamlit as st
import numpy as np
import pandas as pd
import urllib
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instruction.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Explaintory Data Analysis", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Explaintory Data Analysis".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("main.py"))
    elif app_mode == "Explaintory Data Analysis":
        readme_text.empty()
        run_the_app()


def get_file_content_as_string(path):
    
    url = 'https://raw.githubusercontent.com/DaPraxis/IMLdepressionData/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def run_the_app():
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

    # @st.cache(suppress_st_warning=True)
    def scatter_cluster3D(visual, df):
        if(visual=='PCA'):
            model = PCA(n_components=3)
            result = model.fit_transform(df[feat_cols].values)
            st.write('Explained variation per principal component: {}'.format(model.explained_variance_ratio_))
        else:
            model = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
            result = model.fit_transform(df[feat_cols].values)
        df['D1'] = result[:,0]
        df['D2'] = result[:,1] 
        df['D3'] = result[:,2]
        fig = px.scatter_3d(df, x='D1', y='D2', z='D3', color='DRUG')
        st.plotly_chart(fig, use_container_width=True)
    
    # @st.cache(suppress_st_warning=True)
    def scatter_cluster2D(visual, df):
        if(visual=='PCA'):
            model = PCA(n_components=2)
            result = model.fit_transform(df[feat_cols].values)
            st.write('Explained variation per principal component: {}'.format(model.explained_variance_ratio_))
        else:
            model = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
            result = model.fit_transform(df[feat_cols].values)
        df['D1'] = result[:,0]
        df['D2'] = result[:,1] 
        fig = px.scatter(df, x="D1", y="D2", color="DRUG")
        st.plotly_chart(fig, use_container_width=True)

    st.title('EDA')
    st.header('Data Loading')
    filename = file_selector()
    st.write('You selected `%s`' % filename)

    metadata = load_metadata(filename)
    summary = create_summary(metadata)
    showData = st.checkbox('Show raw data')
    if(showData): 
        summary

    desc = {
    'DHAMD1':'DEPRESSED MOOD',
    'DHAMD2':'FEELINGS OF GUILT',
    'DHAMD3':'SUICIDE',
    'DHAMD4':'INSOMNIA: EARLY IN THE NIGHT',
    'DHAMD5':'INSOMNIA: MIDDLE OF THE NIGHT',
    'DHAMD6':'INSOMNIA: EARLY HOURS OF THE MORNING',
    'DHAMD7':'WORK AND ACTIVITIES',
    'DHAMD8':'RETARDATION',
    'DHAMD9':'AGITATION',
    'DHAMD10':'ANXIETY PSYCHIC',
    'DHAMD11':'ANXIETY SOMATIC ',
    'DHAMD12':'SOMATIC SYMPTOMS GASTRO-INTESTINAL',
    'DHAMD13':'GENERAL SOMATIC SYMPTOMS',
    'DHAMD14':'GENITAL SYMPTOMS',
    'DHAMD15':'HYPOCHONDRIASIS',
    'DHAMD16':'LOSS OF WEIGHT',
    'DHAMD17':'INSIGHT'}

    select_from = summary.columns[summary.columns.str.contains('DHAMD')]
    copy = []
    for i, j in enumerate(select_from):
        copy.append( j+': '+desc[j])


    st.header('Data Cluster Presentation')
    check = st.checkbox('Show Scatters')
    hamd = st.sidebar.multiselect('HAMD to invistigate',copy)
    drug = st.sidebar.multiselect('Treatment/Experiment group to compare with', summary.DRUG.unique())
    visual = st.sidebar.selectbox('Cluster Visualization Method', ['PCA', 't-SNE'])
    dimension = st.sidebar.selectbox('Dimension for view', ['3D', '2D'])
    
    df = summary.copy()
    feat_cols = list(map(lambda x: x.split(':')[0], hamd))
    in_scope_drug = list(map(lambda x: x.split(':')[0], drug))
    df = df[pd.DataFrame(df.DRUG.tolist()).isin(in_scope_drug).any(1)]
    # df = df.loc[df.DRUG in in_scope_drug]

    if(len(in_scope_drug)<1 or len(feat_cols)<3):
        st.error('At least with **3 dimensions** and **1 drugs!**')
    else:
        if(not check):
            if(dimension=='3D' and visual=='t-SNE'):
                st.warning('approximately takes {}s'.format(len(hamd)*4+len(drug)*2))
            elif(dimension=='2D' and visual=='t-SNE'):
                st.warning('approximately takes {}s'.format(len(hamd)+len(drug)*2))
            elif(dimension=='3D' and visual=='PCA'):
                st.warning('approximately takes 5s')
            elif(dimension=='2D' and visual=='PCA'):
                st.warning('approximately takes 2s')
    if (dimension == '3D' and len(in_scope_drug)>=1 and len(feat_cols)>=3 and check):
        scatter_cluster3D(visual, df)
    elif (dimension == '2D' and len(in_scope_drug)>=1 and len(feat_cols)>=3 and check):
        scatter_cluster2D(visual, df)

    st.header('Data Density Pair Plots')
    checkPair = st.checkbox('Show Pair Plots')
    pair_feature = feat_cols.copy()
    pair_feature.append('DRUG')
    if(len(in_scope_drug)>0 and len(feat_cols)>0):
        if checkPair:
            state = [len(drug), len(hamd)]
            g = sns.pairplot(df[pair_feature], hue="DRUG")
            st.pyplot()
        else:
            st.warning('approximately takes {}s'.format(len(hamd)*2+len(drug)))
    else:
        st.error('At least with **1 dimension** and **1 drug!**')

if __name__ == "__main__":
    main()