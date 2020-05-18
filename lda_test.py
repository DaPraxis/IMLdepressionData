import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from math import sqrt
from scipy.stats import t

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
    df_w['HAMD_Name'] = ''

    idx = list(df_w.index)
    
    for i in idx:
        
        #add corresponding descriptions
        df_w.loc[i,'Description'] = desc[str(i)]
        df_w.loc[i,'HAMD_Name'] = 'HAM-D '+str(i+1)
        
        #get polarity
        if df_pn.loc[i,'PN'] > 0:
            df_w.loc[i,'Polarity'] = 'positive' 
            df_w.loc[i,'True_Weightings'] = df_w.loc[i,'Weightings']
        else:
            df_w.loc[i,'Polarity'] = 'negative'
            df_w.loc[i,'True_Weightings'] = -1*df_w.loc[i,'Weightings']

    pvals = pd.Series([])
    sigs = pd.Series([])
    ci_up = pd.Series([])
    ci_lo = pd.Series([])
    ci = pd.Series([])
    sig_2 = pd.Series([])
    drug = df.loc[df['DRUG']!=0]
    placebo = df.loc[df['DRUG']==0]
    
    for i in range(len(df_w.Weightings)):
        N1=len(drug[drug.columns[i]])
        N2=len(placebo[placebo.columns[i]])
        d_f = (N1 + N2 - 2)
        std1 = drug[drug.columns[i]].std()
        std2 = placebo[placebo.columns[i]].std()
        std_N1N2 = sqrt( ((N1 - 1)*(std1)**2 + (N2 - 1)*(std2)**2) / d_f)
        diff_mean = drug[drug.columns[i]].mean() - placebo[placebo.columns[i]].mean()
        MoE = t.ppf(0.975, d_f) * std_N1N2 * sqrt(1/N1 + 1/N2)
        ci_up[i] = diff_mean + MoE
        ci_lo[i] = diff_mean - MoE
        sig_2[i] = ((diff_mean + MoE) * (diff_mean - MoE)>0)
        ci[i] = (diff_mean - MoE, diff_mean + MoE)
        tset, pval = ttest_1samp(df_w.Weightings, df_w.Weightings[i])
        sig = ttest_ind(drug[drug.columns[i]], placebo[placebo.columns[i]], equal_var = False)
        pvals[i]=pval
        sigs[i]=sig[1]
    df_w.insert(3, 'weightings p-value', pvals)
    df_w.insert(4, '2-sample t-test p-value', sigs)
    df_w.insert(5, '2-sample t-test ci', ci)
    df_w.insert(6, '2-sample t-test ci upper', ci_up)
    df_w.insert(7, '2-sample t-test ci lower', ci_lo)
    df_w.insert(8, 'Significant', sig_2)
    return (df_w, df)

def overall_test(df):
    color = st.sidebar.selectbox('Colored By', ['Significant', 'Description', 'Polarity'])
    tests = [
        [['Duloxetine'], ['Placebo']],
        [['Fluoxetine'], ['Placebo']],
        [['Paroxetine'], ['Placebo']],
        [['Venlafax'], ['Placebo']],
        [['Escitalopram'], ['Placebo']],
        [['Duloxetine', 'Fluoxetine', 'Paroxetine', 'Venlafax', 'Escitalopram'], ['Placebo']],
        [['Escitalopram', 'Fluoxetine', 'Paroxetine'], ['Placebo']],
        [['Duloxetine', 'Venlafax'], ['Placebo']],
        [['Escitalopram', 'Fluoxetine', 'Paroxetine'],  ['Duloxetine', 'Venlafax']]
    ]

    total_lda_w = []
    for t in tests:
        df_w, df_visual = individual(df.copy(), t[0], t[1])
        total_lda_w.append(df_w)
    df_1 = pd.concat(total_lda_w, keys=['Duloxetine', 'Fluxetine', 'Paroxetine', 'Venlafax', 'Escitalopram', 'Total Drug', 'SSRI', 'SNRI', 'SNRI vs SSRI'])
    df_1.reset_index(inplace=True)
    check = st.checkbox('Analyse!')
    if check:     
        fig = px.bar(df_1,
                    x="True_Weightings",
                    y="HAMD_Name",
                    animation_frame="level_0",
                    orientation='h',
                    range_x=[-0.5, 0.5],
                    color=color,
                    hover_name="Description",
                    hover_data=["2-sample t-test p-value", "weightings p-value", '2-sample t-test ci'],
                    error_x = '2-sample t-test ci upper',
                    error_x_minus = '2-sample t-test ci lower'
                    )
        fig.update_layout(width=1140,
                    height=800,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    title_text='LDA Hamilton Weightings',
                    showlegend=True)
        title = 'Anti-Depression Drugs Overview'
        fig.update_xaxes(title_text=title)
        fig.update_yaxes(title_text='')
        
        st.plotly_chart(fig, use_container_width=True)
    return

def custom_test(df):
    g1 = st.sidebar.multiselect('Group1', df.DRUG.unique().tolist())
    g2 = st.sidebar.multiselect('Group2', df.DRUG.unique().tolist())
    color = st.sidebar.selectbox('Colored By', ['Significant','Polarity', 'Description'])
    if(len(g1)<1 or len(g2)<1):
        st.error('At least two groups to compare')
    else:
        check = st.checkbox('Analyse!')
        if check:     
            df_w, df_visual = individual(df.copy(), g1, g2)
            
            fig = px.bar(df_w,
                        x="True_Weightings",
                        y="Description",
                        # animation_frame="level_0",
                        orientation='h',
                        range_x=[-0.5, 0.5],
                        color=color,
                        hover_name="Description",
                        hover_data=["2-sample t-test p-value", "weightings p-value", '2-sample t-test ci'],
                        error_x = '2-sample t-test ci upper',
                        error_x_minus = '2-sample t-test ci lower')
            fig.update_layout(width=1140,
                        height=800,
                        xaxis_showgrid=False,
                        yaxis_showgrid=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        title_text='LDA Hamilton Weightings',
                        showlegend=True)
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

def lda_test(df):
    df = df.copy()
    select = st.selectbox('View Type', ['Overview', 'Custom'])
    switcher = {
        'Overview':overall_test,
        'Custom': custom_test
    }
    func = switcher.get(select, lambda: "Invalid Method")
    func(df)
    return