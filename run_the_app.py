import streamlit as st
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px

from dataloading_process import get_file_content_as_string, load_metadata, create_summary, file_selector

from lda_test import lda_test



def run_the_app():
    st.title('Unsupervised Learning for HAM-D Scores vs. Drugs')
    st.markdown('👈 This sections, we only learns from [***the Hamilton Depression Rating Scale (HAM-D)***](https://www.mdcalc.com/hamilton-depression-rating-scale-ham-d) attributes')
    st.header('Data Loading')
    filename = file_selector()
    st.write('You selected `%s`' % filename)

    metadata = load_metadata(filename)
    summary = create_summary(metadata)

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


    st.header('Data Visualization')
    allMethods = ['Clustering Visualization', 'LDA Analysis', 'Density Plot']
    method = st.selectbox('Select Visualization', allMethods)

    def cluster():
        st.header('Data Clustering Visualization')
        check = st.checkbox('Show Scatters')
        hamd = st.sidebar.multiselect('HAMD to invistigate',copy)
        drug = st.sidebar.multiselect('Treatment/Experiment group to compare with', summary.DRUG.unique())
        visual = st.sidebar.selectbox('Cluster Visualization Method', ['PCA', 't-SNE'])
        if(visual=='t-SNE'):
            perplexity = st.sidebar.slider('Select Perplexity for t-SNE (number of nearest neighbors)', 5.0, 70.0, 10.0)
            iters = st.sidebar.slider('Select Number of Training Iterations for t-SNE', 250, 2000, 1000)
            lr = st.sidebar.slider('Select Learning Rate for t-SNE', 10, 1000, 200)
            init = st.sidebar.selectbox('Select Initialization Embadding', ['random', 'pca'])
        dimension = st.sidebar.selectbox('Dimension for view', ['3D', '2D'])
        df = summary.copy()
        feat_cols = list(map(lambda x: x.split(':')[0], hamd))
        in_scope_drug = list(map(lambda x: x.split(':')[0], drug))
        df = df[pd.DataFrame(df.DRUG.tolist()).isin(in_scope_drug).any(1)]
        # df = df.loc[df.DRUG in in_scope_drug]

        # @st.cache(hash_funcs={px.scatter_3d: (feat_cols, visual, in_scope_drug)},suppress_st_warning=True)
        def scatter_cluster3D(visual, df, feat_cols, in_scope_drug):
            if(visual=='PCA'):
                model = PCA(n_components=3)
                result = model.fit_transform(df[feat_cols].values)
                st.write('Explained variation per principal component: {}'.format(model.explained_variance_ratio_))
            else:
                model = TSNE(n_components=3, verbose=1, perplexity=perplexity, n_iter = iters, learning_rate = lr, init=init)
                result = model.fit_transform(df[feat_cols].values)
            df['D1'] = result[:,0]
            df['D2'] = result[:,1] 
            df['D3'] = result[:,2]
            fig = px.scatter_3d(df, x='D1', y='D2', z='D3', color='DRUG')
            st.plotly_chart(fig, use_container_width=True)
            return fig
        
        # @st.cache(suppress_st_warning=True)
        def scatter_cluster2D(visual, df, feat_cols, in_scope_drug):
            if(visual=='PCA'):
                model = PCA(n_components=2)
                result = model.fit_transform(df[feat_cols].values)
                st.write('Explained variation per principal component: {}'.format(model.explained_variance_ratio_))
            else:
                model = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter = iters, learning_rate = lr, init=init)
                # n_components=2, verbose=1, random_state=0, angle=.99, init='pca'
                result = model.fit_transform(df[feat_cols].values)
            df['D1'] = result[:,0]
            df['D2'] = result[:,1] 
            fig = px.scatter(df, x="D1", y="D2", color="DRUG")
            st.plotly_chart(fig, use_container_width=True)
            return fig

        if(len(in_scope_drug)<1 or len(feat_cols)<3):
            st.error('At least with **3 dimensions** and **1 drugs!**')
        else:
            if(not check):
                if(dimension=='3D' and visual=='t-SNE'):
                    upper = (len(hamd)*4+len(drug)*2)*200/lr*iters/1000*perplexity/10
                    lower = (len(hamd)*8+len(drug)*4)*200/lr*iters/1000*perplexity/10
                    st.warning('approximately takes {0:,.2f}s to {1:,.2f}s'.format(upper, lower))
                elif(dimension=='2D' and visual=='t-SNE'):
                    upper = (len(hamd)+len(drug)*2)*200/lr*iters/1000*perplexity/10
                    lower = (len(hamd)*2+len(drug)*4)*200/lr*iters/1000*perplexity/10
                    st.warning('approximately takes {0:,.2f}s to {1:,.2f}s'.format(upper, lower))
                elif(dimension=='3D' and visual=='PCA'):
                    st.warning('approximately takes 5s')
                elif(dimension=='2D' and visual=='PCA'):
                    st.warning('approximately takes 2s')
        if (dimension == '3D' and len(in_scope_drug)>=1 and len(feat_cols)>=3 and check):
            fig = scatter_cluster3D(visual, df, feat_cols, in_scope_drug)
        elif (dimension == '2D' and len(in_scope_drug)>=1 and len(feat_cols)>=3 and check):
            fig = scatter_cluster2D(visual, df, feat_cols, in_scope_drug)
        return 

    def pairPlot():
        hamd = st.sidebar.multiselect('HAMD to invistigate',copy)
        drug = st.sidebar.multiselect('Treatment/Experiment group to compare with', summary.DRUG.unique())
        df = summary.copy()
        feat_cols = list(map(lambda x: x.split(':')[0], hamd))
        in_scope_drug = list(map(lambda x: x.split(':')[0], drug))
        df = df[pd.DataFrame(df.DRUG.tolist()).isin(in_scope_drug).any(1)]
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
        return

    def lda():
        # df = summary.copy()
        lda_test(summary)
    
    def selectMethods(arg):
        switcher = {
            'Clustering Visualization': cluster,
            'Density Plot': pairPlot,
            'LDA Analysis': lda
        }
        func = switcher.get(arg, lambda: "Invalid Method")
        func()
    
    selectMethods(method)