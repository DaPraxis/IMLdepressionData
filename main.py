import streamlit as st

from run_the_app import run_the_app
from dataloading_process import get_file_content_as_string
from cluster_learn import cluster_learn


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instruction.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Explaintory Data Analysis", 'Unsupervised Learning', "Show the source code"])
    st.sidebar.title('Model')
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Explaintory Data Analysis".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("main.py"))
    elif app_mode == "Explaintory Data Analysis":
        readme_text.empty()
        run_the_app()
    elif app_mode == 'Unsupervised Learning':
        cluster_learn()


if __name__ == "__main__":
    main()