import streamlit as st
import numpy as np
import pandas as pd
import missingno as msno

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from dataloading_process import get_file_content_as_string, load_metadata, file_selector
def eda():
    st.title('EDA')

    st.header('Data Loading')
    filename = file_selector()
    st.write('You selected `%s`' % filename)
    df = load_metadata(filename)

    # df for missing data and percentage
    nans = pd.concat([df.isnull().sum(), (df.isnull().sum() / df.shape[0]) * 100], axis=1, keys=['Num_NaN', 'NaN_Percent'])
    nans = nans.sort_values(by=['NaN_Percent'], ascending=False)

    def catd():
        @st.cache
        def select_data():
            return df.select_dtypes(include=['object'])
        df_cat = select_data()
        if(df_cat.empty):
            st.error('No Categorical Data')
        else:
            bar = st.progress(0)
            for i, c in enumerate(df_cat.columns.tolist()):
                check_plot = st.checkbox(c)
                bar.progress((i+1)/len(df_cat.columns.tolist()))
                if(check_plot):
                    fig1 = px.histogram(df_cat, x=c)
                    st.plotly_chart(fig1)
                    if(len(df_cat[c].unique().tolist())<=20):
                        freq = pd.DataFrame(df_cat[c].value_counts())
                        fig2 = px.pie(df_cat, values=freq[c], names=freq.index.tolist(), title=c)
                        st.plotly_chart(fig2)

    def boold():
        @st.cache
        def select_data():
            return df.select_dtypes(include='bool')
        df_bool = select_data()
        if(df_bool.empty):
            st.error('No Boolean Data')

    def numd():
        @st.cache
        def select_data():
            return df.select_dtypes(include='number')
        df_num = select_data()
        if(df_num.empty):
            st.error('No Numerical Data')
        else:
            bar = st.progress(0)
            for i, c in enumerate(df_num.columns.tolist()):
                check_plot = st.checkbox(c)
                bar.progress((i+1)/len(df_num.columns.tolist()))
                if(check_plot):
                    fig2 = px.histogram(df_num, x = c, marginal='box')
                    st.plotly_chart(fig2)
                    st.warning('Missing values are droped for density plot')
                    fig3 = ff.create_distplot([df_num[c].dropna()], [c], show_hist=False)
                    st.plotly_chart(fig3)

    def dated():
        @st.cache
        def select_data():
            return df.select_dtypes(include='datetime')
        df_date = select_data()
        if(df_date.empty):
            st.error('No Date Time Data')
        return


    def selectDataType():
        selected = st.selectbox('Select Data Types', ['Categorical', 'Numeric', 'Boolean', 'Data Time'])
        switcher = {
            'Categorical': catd,
            'Numeric': numd,
            'Boolean': boold,
            'Data Time': dated
        }
        func = switcher.get(selected, lambda: "Invalid Method")
        func()

    def ms_over():
        ranges = st.sidebar.slider('Range of Features to View', 0.0, 100.0, (0.0, 25.0))
        cols_with_nans = nans.index.tolist()
        col_ranges = cols_with_nans[int(len(cols_with_nans)*ranges[0]/100):int(len(cols_with_nans)*ranges[1]/100)]
        df_select = df[col_ranges]
        msno.matrix(df=df_select, figsize=(30, 15), color=(0.24, 0.77, 0.77))
        st.pyplot()
        st.table(nans.loc[col_ranges])

    def only_ms():
        labels = []
        ranges = st.sidebar.slider('Range of Missing% to View', 0.0, 100.0, (50.0, 70.0))
        for i in nans.index:
            if(nans['NaN_Percent'][i]<=ranges[1] and nans['NaN_Percent'][i]>=ranges[0]):
                labels.append(i)
            elif(nans['NaN_Percent'][i]<ranges[1]):
                break
        df_select = df[labels]
        msno.matrix(df=df_select, figsize=(30, 15), color=(0.24, 0.77, 0.77))
        st.pyplot()
        st.table(nans.loc[labels])
        return

    def selectMissingValues():
        selected = st.selectbox('Missing Value Visualizations', ['By Feature', 'By Missing%'])
        switcher = {
            'By Feature': ms_over,
            'By Missing%': only_ms
        }
        func = switcher.get(selected, lambda: "Invalid Method")
        func()
        return

    method = st.sidebar.selectbox('Select EDA View', ['By Data Type', 'By Misssing Values'])

    def selectMethod(arg):
        switcher = {
            'By Data Type': selectDataType,
            'By Misssing Values': selectMissingValues,
        }
        func = switcher.get(arg, lambda: "Invalid Method")
        func()

    selectMethod(method)

    return