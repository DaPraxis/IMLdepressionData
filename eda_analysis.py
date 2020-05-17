import streamlit as st
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dataloading_process import get_file_content_as_string, load_metadata, file_selector
def eda():
    st.title('EDA')

    st.header('Data Loading')
    filename = file_selector()
    st.write('You selected `%s`' % filename)
    df = load_metadata(filename)

    selected = st.selectbox('Select Data Types', ['Categorical', 'Number', 'Boolean', 'Data Time'])

    def catd():
        @st.cache
        def select_data():
            return df.select_dtypes(include=['object', 'category'])
        df_cat = select_data()
        if(df_cat.empty):
            st.error('No Categorical Data')
        else:
            for i, c in enumerate(df_cat.columns.tolist()):
                fig = px.histogram(df, x=c)
                st.plotly_chart(fig)

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

    def dated():
        @st.cache
        def select_data():
            return df.select_dtypes(include='datetime')
        df_date = select_data()
        if(df_date.empty):
            st.error('No Date Time Data')
        return


    def selectMethods(arg):
        switcher = {
            'Categorical': catd,
            'Number': numd,
            'Boolean': boold,
            'Data Time': dated
        }
        func = switcher.get(arg, lambda: "Invalid Method")
        func()
    selectMethods(selected)

    return