from __future__ import annotations
import streamlit as st
import pandas as pd
import json

st.set_page_config(layout='wide')

class App:

    with open('strings.txt', 'r') as f:
        __STRINGS = json.load(f)

    def __init__(self):
        # Preset session states; Used to preserve states on reruns/button pushes
        if 'data' not in st.session_state:
            st.session_state.data = ""
        if 'has_data' not in st.session_state:
            st.session_state.has_data = False
        if 'raw_toggle' not in st.session_state:
            st.session_state.raw_toggle = False

        return
    
    def __to_state_false(self, 
                     key: str):
        st.session_state[key] = False
    
    
    def __load_data(self,
                    file: st.UploadedFile):
        
        @st.cache_data
        def cache_data(file):
            return pd.read_csv(file)

        has_data = False
        data = ""
        error = ""

        if file is None:
            error = self.__STRINGS['Data_Error_Empty']
        elif file.name.split('.')[-1] != 'csv':
            error = self.__STRINGS['Data_Error_Type']

        if error == "":
            data = cache_data(file)

            if len(data) == 0:
                data = ""
                error = self.__STRINGS['Data_Error_Empty']

        
        if error == "":
            has_data = True

        return has_data, data, error
        
    def window(self):
        # Title
        st.title(self.__STRINGS['Title'])

        

        col1, col2 = st.columns([1.5,2])

        # col1: "Form"
        # - Target Column: target_input
        # - ID Column: id_input
        # - Classifier Selector: classifier_input
        # - Cluster Selector: cluster_input
        # - Data Upload: data_input
        # - Run Button
        # - View Raw Data

        # Data
        data_file = col1.file_uploader(self.__STRINGS['Data_Entry'], 
                                       type = 'csv', 
                                       accept_multiple_files = False,
                                       on_change=self.__to_state_false,
                                       args=['raw_toggle'],
                                       key = 'data_input')
        data_err = col1.empty()

        has_data, data, data_err_msg = self.__load_data(data_file)
        st.session_state.has_data = has_data
        st.session_state.data = data
            

        if data_err_msg != "":
            data_err.error(data_err_msg)

        cols = self.__STRINGS['Default_Selectbox']
        if st.session_state.has_data:
            cols.extend(list(data.columns))

        # Target column and error message
        target_col = col1.selectbox(self.__STRINGS['Target_Entry'],
                                    options = cols.copy(),
                                    key = 'target_input')
        target_err = col1.empty()
        
        if (st.session_state.has_data) and (target_col == self.__STRINGS['Default_Selectbox'][0]):
            target_err.error(self.__STRINGS['Column_Error_Empty'])
        

        cols.insert(1,self.__STRINGS['None_Selectbox_Option'])
        # ID column and error message
        id_col = col1.selectbox(self.__STRINGS['ID_Entry'],
                                options = cols.copy(),
                                key = 'id_input')
        id_err = col1.empty()

        if (st.session_state.has_data) and (id_col == self.__STRINGS['Default_Selectbox'][0]):
            id_err.error(self.__STRINGS['Column_Error_Empty'])
        if (st.session_state.has_data) and (id_col == target_col):
            id_err.error(self.__STRINGS['Column_Error_Dup'])

        # Classifier Selector
        classifier = col1.selectbox('Choose a classifier:', 
                                    options = self.__STRINGS['Classifier_Options'], 
                                    key = 'classifier_input')

        raw_data = col1.toggle(self.__STRINGS['Show_Data_Toggle'],
                               key = 'raw_toggle',
                               disabled = not st.session_state.has_data)

        if raw_data:
            col1.dataframe(st.session_state.data)

        ###### TEMP ######
        col2.write(st.session_state)




if __name__ == '__main__':
    app = App()
    app.window()