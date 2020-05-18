import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

import plotly.express as px

import streamlit as st

desc = {
    '0':'DEPRESSED MOOD',
    '1':'FEELINGS OF GUILT',
    '2':'SUICIDE',
    '3':'INSOMNIA: EARLY IN THE NIGHT',
    '4':'INSOMNIA: MIDDLE OF THE NIGHT',
    '5':'INSOMNIA: EARLY HOURS OF THE MORNING',
    '6':'WORK AND ACTIVITIES',
    '7':'RETARDATION',
    '8':'AGITATION',
    '9':'ANXIETY PSYCHIC',
    '10':'ANXIETY SOMATIC ',
    '11':'SOMATIC SYMPTOMS GASTRO-INTESTINAL',
    '12':'GENERAL SOMATIC SYMPTOMS',
    '13':'GENITAL SYMPTOMS',
    '14':'HYPOCHONDRIASIS',
    '15':'LOSS OF WEIGHT',
    '16':'INSIGHT'} 

def individual(df, label1, label2):
    '''
      df: first-last (17 features with DHAMD scores and 1 column of DRUG names)
      drug_name: name of drug you want to compare with Placebo
    '''

    df['DRUG'].replace(label1, 1, inplace=True)

    df['DRUG'].replace(label2, 0, inplace=True)

    df = df[(df['DRUG']==1) | (df['DRUG']==0)].reset_index(drop=True)

    #fit LDA model
    y = df['DRUG'].astype(int)
    X = df.drop(['DRUG'],axis=1)

    
    clf = lda()
    model = clf.fit(X,y)
    df_pn = pd.DataFrame(model.coef_,index=['PN']).T

    #get weightings
    df_w = pd.DataFrame(np.absolute(model.coef_),index=['Weightings']).sort_values(by='Weightings', axis=1,ascending=False).T
    
    #sort values and add descriptions
    df_w['Polarity'] = ''
    df_w['Description'] = ''
    df_w['True_Weightings'] = ''
    idx = list(df_w.index)
    
    for i in idx:
        
        #add corresponding descriptions
        df_w.loc[i,'Description'] = desc[str(i)]
        
        #get polarity
        if df_pn.loc[i,'PN'] > 0:
            df_w.loc[i,'Polarity'] = 'positive' 
            df_w.loc[i,'True_Weightings'] = df_w.loc[i,'Weightings']
        else:
            df_w.loc[i,'Polarity'] = 'negative'
            df_w.loc[i,'True_Weightings'] = -1*df_w.loc[i,'Weightings']

    # t-test
    print('t-test')
    pvals = pd.Series([])
    sigs = pd.Series([])
    drug = df.loc[df['DRUG']!='Placebo']
    placebo = df.loc[df['DRUG']=='Placebo']
    for i in range(len(df_w.Weightings)):
        tset, pval = ttest_1samp(df_w.Weightings, df_w.Weightings[i])
        sig = ttest_ind(drug[drug.columns[i]], placebo[placebo.columns[i]])
        pvals[i]=pval
        sigs[i]=sig
    df_w.insert(3, 'p-value', pvals)
    df_w.insert(4, 'ci', sigs)
    df_w['Significant'] = df_w['p-value'] <=0.05
    return (df_w, df)



def lda_test(df):
    g1 = st.sidebar.multiselect('Group1', df.DRUG.unique().tolist())
    g2 = st.sidebar.multiselect('Group2', df.DRUG.unique().tolist())
    if(len(g1)<1 or len(g2)<1):
        st.error('At least two groups to compare')
    else:
        check = st.checkbox('Analyse!')
        if check:     
            df_w, df_visual = individual(df, g1, g2)
            
            fig = px.bar(df_w,
                        x="True_Weightings",
                        y="Description",
                        # animation_frame="level_0",
                        orientation='h',
                        range_x=[-0.5, 0.5],
                        color="Description")
            fig.update_layout(width=1140,
                        height=800,
                        xaxis_showgrid=False,
                        yaxis_showgrid=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title_text='LDA Hamilton Weightings',
                        showlegend=False)
            title = 'Anti-Depression Drugs'
            for i in g1:
                title+=' '+i
            title+=' vs.'
            for j in g2:
                title+=' '+j
            fig.update_xaxes(title_text=title)
            fig.update_yaxes(title_text='')
            st.plotly_chart(fig, use_container_width=True)
    return