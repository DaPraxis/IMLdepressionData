import streamlit as st
import numpy as np
import pandas as pd
import urllib


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instruction.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("main.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        # run_the_app()


def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/dapraxis/IMLdepressionData/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

if __name__ == "__main__":
    main()