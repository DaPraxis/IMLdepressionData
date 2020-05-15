import streamlit as st
import numpy as np
import pandas as pd
import urllib
import os


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
        run_the_app()


def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/dapraxis/IMLdepressionData/dataset/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

# def frame_selector_ui(summary):
#     st.sidebar.markdown("# Frame")

#     # The user can pick which type of object to search for.
#     object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)

#     # The user can select a range for how many of the selected objecgt should be present.
#     min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % object_type, 0, 25, [10, 20])
#     selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
#     if len(selected_frames) < 1:
#         return None, None

#     # Choose a frame out of the selected frames.
#     selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)

#     # Draw an altair chart in the sidebar with information on the frame.
#     objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
#     chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
#         alt.X("index:Q", scale=alt.Scale(nice=False)),
#         alt.Y("%s:Q" % object_type))
#     selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
#     vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(
#         alt.X("selected_frame:Q", axis=None)
#     )
#     st.sidebar.altair_chart(alt.layer(chart, vline))

#     selected_frame = selected_frames[selected_frame_index]
#     return selected_frame_index, selected_frame

def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
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

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    def file_selector(folder_path='.'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file to load data', filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()

    st.write('You selected `%s`' % filename)

    metadata = load_metadata(filename)
    summary = create_summary(metadata)
    summary
    

if __name__ == "__main__":
    main()