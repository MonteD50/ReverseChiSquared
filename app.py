import streamlit as st
import pandas as pd
import io, base64
from createData import Main
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle   

st.title('Data Generation Using Reverse Chi- Squared Technique')

fileCSV = st.file_uploader("Upload a csv file", type=([".csv"]))

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

if fileCSV:
    uploadedFile = io.TextIOWrapper(fileCSV)
    df = pd.read_csv(uploadedFile)

    st.write('Your Dataset')
    st.write(df)

    columns = st.sidebar.multiselect('Which columns do you want to generate more of?', df.columns)

    if len(columns) != 0:
        numSubsetAmount = st.sidebar.slider('Percentage to split (train- test) to use in creation (the rest will be used in testing)', 0.0, 1.0, 0.5)
        
        # TODO: number to generate is incorrect output
        numSamplesToGenerate = st.sidebar.number_input('How many new samples would you like to generate? (Warning: If this is too large, samples will repeat). (note: This is an approximate amount)', 0, value=100)
        numNewSamplesToUse = int(numSamplesToGenerate / len(df)) + 1

        if numSubsetAmount != 0:
            numSubset = int((1 - numSubsetAmount) * len(df))

            subset_df = df.sample(numSubset)

            ins = subset_df.index.tolist()

            testing_df = df.loc[df.index.isin(ins) == False]
            
            st.write('Subset Dataset (what will be used for creating new data)')
            st.write(subset_df[columns])

            st.write('Testing Dataset (what should be used for testing)')
            st.write(testing_df[columns])

            if st.button('Download Testing Subset Data as CSV'):
                downloadLink = download_link(testing_df, 'testing.csv', 'Click here to download your data')
                st.markdown(downloadLink, unsafe_allow_html=True)
        else:
            subset_df = df

        numCategories = 150

        main = Main(subset_df, columns, numNewSamplesToUse, numCategories)
        result = main.run()

        st.write('Generated Data (combined with data used to generate data)')

        st.write('Generated ' + str(len(result) - len(subset_df)) + ' new values.')
        
        st.write(result)

        if st.button('Download Generated Data as CSV'):
            downloadLink = download_link(result, 'generated.csv', 'Click here to download your data')
            st.markdown(downloadLink, unsafe_allow_html=True)

        explore = st.sidebar.checkbox('Explore generated data')
        if explore:
            # TODO: explore generted and og data if explore selected
            #st.write('Line Chart of ')

            st.write('Histogram of Distribution between Generated and Testing Data')
            st.write('Note: If you get a distribution not similiar to the test generation, rerun the file as this most likely is due to an unlucky random sample of the subset and testing dataset')
            histColumn = st.selectbox('Which columns do you want to compare in a histogram?', result.columns, index =0)

            if len(histColumn) != 0:
                fig, ax = plt.subplots()
                ax.hist(result[histColumn], color='orange')
                ax.hist(testing_df[histColumn], color='blue')

                handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['orange', 'blue']]
                labels= ["Generated Dataset", "Testing Dataset"]
                fig.legend(handles, labels)

                st.pyplot(fig)

    st.write("Warning: Percentage values work best as this was initially created for that. However, better support for discrete, non- percentage like columns will be added soon.")