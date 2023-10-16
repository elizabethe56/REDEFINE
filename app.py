from __future__ import annotations
import streamlit as st
import pandas as pd
import json

from redefine import REDEFINE

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

        if 'target_valid' not in st.session_state:
            st.session_state.target_valid = False
        if 'id_valid' not in st.session_state:
            st.session_state.id_valid = False
        
        if 'data_set' not in st.session_state:
            st.session_state.data_set = False

        if 'param_dict' not in st.session_state:
            st.session_state.param_dict = {}

        if 'raw_toggle' not in st.session_state:
            st.session_state.raw_toggle = False

        return
    
    def __to_state_false(self,
                         key: str | list):
        if type(key) == str:
            st.session_state[key] = False
        else:
            for k in key:
                st.session_state[k] = False
    
    def __load_data(self,
                    file: st.UploadedFile):
        
        @st.cache_data
        def verify_data(file):
            has_data = False
            data = ""
            error = ""

            if file is None:
                error = self.__STRINGS['Data_Error_Empty']
            elif file.name.split('.')[-1] != 'csv':
                error = self.__STRINGS['Data_Error_Type']

            if error == "":
                data = pd.read_csv(file)

                if len(data) == 0:
                    data = ""
                    error = self.__STRINGS['Data_Error_File_Empty']
                
            if error == "":
                has_data = True

            return has_data, data, error
        return verify_data(file)
    
    def __set_data(self):
        
        @st.cache_data
        def create_redefine_object(data, target_col, id_col):
            return REDEFINE(data, target_col, id_col)
        
        # If data and columns are valid:
        if st.session_state.has_data and st.session_state.target_valid and st.session_state.id_valid:
            try:
                st.session_state.redefine = create_redefine_object(st.session_state.data.copy(),
                                                                   st.session_state.target_input,
                                                                   st.session_state.id_input)
                st.session_state.data_set = True
            except:
                return
        return
        
    def window(self):
        # Title
        st.title(self.__STRINGS['Title'])
        st.header(self.__STRINGS['Subtitle'])
        
        col1, col2 = st.columns([1,1])

        # col1: "Form"
        # - Target Column: target_input
        # - ID Column: id_input
        # - Classifier Selector: classifier_input
        # - Cluster Selector: cluster_input
        # - Data Upload: data_input
        # - Run Button
        # - View Raw Data

        col1.subheader(self.__STRINGS['Header1'])

        ########## Data ##########
        data_file = col1.file_uploader(self.__STRINGS['Data_Entry'], 
                                       type = 'csv', 
                                       accept_multiple_files = False,
                                       on_change=self.__to_state_false,
                                       args=[['raw_toggle', 'data_set']],
                                       key = 'data_input')

        has_data, data, data_err_msg = self.__load_data(data_file)
        st.session_state.has_data = has_data
        st.session_state.data = data
        
        # Data error
        if data_err_msg != "":
            col1.error(data_err_msg)

        # Retrieve column names for selectboxes
        cols = self.__STRINGS['Default_Selectbox']
        if st.session_state.has_data:
            cols.extend(list(data.columns))

        ########## Target column ##########
        target_col = col1.selectbox(self.__STRINGS['Target_Entry'],
                                    options = cols.copy(),
                                    key = 'target_input',
                                    on_change = self.__to_state_false,
                                    args = ['target_valid'])

        if st.session_state.has_data:
            if target_col == self.__STRINGS['Default_Selectbox'][0]:
                col1.error(self.__STRINGS['Column_Error_Empty'])
            else:
                st.session_state.target_valid = True

        ########## ID column and error message ##########
        cols.insert(1, self.__STRINGS['None_Selectbox_Option'])
        id_col = col1.selectbox(self.__STRINGS['ID_Entry'],
                                options = cols.copy(),
                                key = 'id_input',
                                on_change = self.__to_state_false,
                                args = ['id_valid'])

        if st.session_state.has_data:
            if id_col == self.__STRINGS['Default_Selectbox'][0]:
                col1.error(self.__STRINGS['Column_Error_Empty'])
            elif id_col == target_col:
                col1.error(self.__STRINGS['Column_Error_Dup'])
            else:
                st.session_state.id_valid = True

        ########## Set Data Button ##########
        set_data = col1.button(self.__STRINGS['Set_Data_Button'],
                                  key = 'set_data_button',
                                  on_click = self.__set_data)

        # Only show Classifier/Cluster options if data and columns are valid
        if st.session_state.data_set:
            ########## Classifier Selector ##########
            col1.markdown('-----')
            col1.subheader(self.__STRINGS['Header2'])

            classifier = col1.selectbox(self.__STRINGS['Classifier_Entry'], 
                                        options = self.__STRINGS['Classifier_Options'], 
                                        key = 'classifier_input')
            
            subcol1, subcol2, subcol3, subcol4 = col1.columns(4)

            # columns for parameter inputs
            params = [subcol1.empty(), subcol2.empty(), subcol3.empty(), subcol4.empty()]
            validate = subcol1.empty()


            if classifier != self.__STRINGS['Default_Selectbox'][0]:
                param_inputs = {}
                # generate each parameter text input and collect their values
                for i in range(len(self.__STRINGS["Classifier_Params"][classifier])):
                    param_name = self.__STRINGS["Classifier_Params"][classifier][i]
                    # Set default value to previous entry, otherwise empty
                    try:
                        value = st.session_state.param_dict[classifier][param_name]
                    except:
                        value = None
                    
                    temp = params[i].text_input(param_name, value)
                    param_inputs[param_name] = temp

                st.session_state.param_dict[classifier] = param_inputs
                
                validate.button('Validate')
            
            # Cluster Selector
            cluster = col1.selectbox(self.__STRINGS['Cluster_Entry'], 
                                        options = self.__STRINGS['Cluster_Options'], 
                                        key = 'cluster_input')

        # Raw Data Toggle
        raw_data = col2.toggle(self.__STRINGS['Show_Data_Toggle'],
                               key = 'raw_toggle',
                               disabled = not st.session_state.has_data)

        if raw_data:
            col2.dataframe(st.session_state.data)
            if st.session_state.data_set:
                col2.dataframe(st.session_state.redefine.get_X())

        ###### TEMP ######
        col2.write(st.session_state)




if __name__ == '__main__':
    app = App()
    app.window()