from __future__ import annotations
import streamlit as st
import pandas as pd
import json
import os

from redefine import REDEFINE

st.set_page_config(layout='wide')

# TODO: add tooltips/help to UI inputs

class App:

    with open('strings.txt', 'r') as f:
        __STRINGS = json.load(f)

    __DEMO_DATA_PATH = './data/iris_modified.csv'

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
        
        if 'class_results_err' not in st.session_state:
            st.session_state.class_results_err = None
        if 'class_results' not in st.session_state:
            st.session_state.class_results = None
        
        if 'clust_results_err' not in st.session_state:
            st.session_state.clust_results_err = None
        if 'clust_results' not in st.session_state:
            st.session_state.clust_results = None

        if 'run_err' not in st.session_state:
            st.session_state.run_err = None
        if 'run_results' not in st.session_state:
            st.session_state.run_results = None

        if 'raw_toggle' not in st.session_state:
            st.session_state.raw_toggle = False

        return
    
    def __to_state_false(self,
                         key: str | list[str]):
        '''
        Turns the given session state(s) false.  
        Used in on_click to reset states or turn functionalities off
        
        Input:
            key: string or list of strings with the names of session states
        '''

        if type(key) == str:
            st.session_state[key] = False
        else:
            for k in key:
                st.session_state[k] = False
        return
    
    def __load_data(self,
                    file: st.UploadedFile | str
                    ) -> (bool, pd.DataFrame, str):
        '''
        Checks the data for errors and loads the data.
        Uses an internal function to cache the data
        
        Input:
            file: file via st.file_uploader | str if using demo data
        Output:
            has_data: boolean for has_data flag
            data: pandas DataFrame of data if verified
            error: error string, if found
        '''

        @st.cache_data
        def verify_data(file):
            has_data = False
            data = ""
            error = ""

            if type(file) != str:
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
        '''
        Initiates REDFINE object with the data if all data and column specifications are all valid.
        Triggered with the set_data_button.  If data gets set, the 
        Uses an internal function to cache the data.
        '''
        
        @st.cache_data
        def create_redefine_object(file_name, data, target_col, id_col):
            return REDEFINE(file_name, data, target_col, id_col)
        
        # If data and columns are valid:
        if st.session_state.has_data and st.session_state.target_valid and st.session_state.id_valid:
            if st.session_state.demo_toggle:
                file_name = os.path.basename(self.__DEMO_DATA_PATH)
            else:
                file_name = st.session_path.data_input['name']

            try:
                st.session_state.redefine = create_redefine_object(file_name,
                                                                   st.session_state.data.copy(),
                                                                   st.session_state.target_input,
                                                                   st.session_state.id_input)
                st.session_state.data_set = True
            except Exception as e:
                print(e)
                return
        return
    
    def __validate_class_clust(self, model, model_type):
        '''
        Runs a 10-fold cross validation with the chosen classifier model and hyperparameters.
        Inputs:
            classifier: name of classifier model
        '''
        info = st.session_state.redefine.run_model(model, 
                                                   st.session_state.param_dict[model],
                                                   st.session_state.scaler_input,
                                                   model_type)
        if model_type == 'classifier':
            st.session_state.class_results_err = info['error']
            st.session_state.class_results = info['score']
        elif model_type == 'cluster':
            st.session_state.clust_results_err = info['error']
            st.session_state.clust_results = info['score']
        return
    
    def __run(self):
        print("Run!")
        classifier = st.session_state.classifier_input
        cluster = st.session_state.cluster_input
        err, results = st.session_state.redefine.run_redefine(classifier,
                                                        st.session_state.param_dict[classifier],
                                                        cluster,
                                                        st.session_state.param_dict[cluster],
                                                        st.session_state.scaler_input)
        
        st.session_state.run_err = err
        st.session_state.run_results = results
        return
    
    def __test_func(self, loc):
        loc.write("SUCCESS!")
        
    def window(self):
        '''
        Main body for the UI.
        '''

        # Titles
        st.title(self.__STRINGS['Title'])
        st.header(self.__STRINGS['Subtitle'])
        
        col1, col2 = st.columns([1,1])

        col1.subheader(self.__STRINGS['Header1'])
        # Data Input
        # region
        ########## Demo Data ##########
        demo_data = col1.toggle(self.__STRINGS['Data_Demo'],
                               key = 'demo_toggle',
                               on_change = self.__to_state_false,
                               args = [['raw_toggle', 'data_set']])

        ########## Data ##########
        data_file = col1.file_uploader(self.__STRINGS['Data_Entry'], 
                                       type = 'csv', 
                                       accept_multiple_files = False,
                                       key = 'data_input',
                                       on_change=self.__to_state_false,
                                       args=[['raw_toggle', 'data_set']],
                                       disabled = st.session_state.demo_toggle)

        if demo_data:
            has_data, data, data_err_msg = self.__load_data(self.__DEMO_DATA_PATH)
            st.session_state.target_input = self.__STRINGS['Data_Demo_Cols']['target']
            st.session_state.id_input = self.__STRINGS['Data_Demo_Cols']['id']
        else:
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
                                    args = ['target_valid'],
                                    disabled = st.session_state.demo_toggle)
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
                                args = ['id_valid'],
                                disabled = st.session_state.demo_toggle)

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
        # endregion
        
        # Model Selection
        # region
        # Only show Classifier/Cluster options if data and columns are valid
        if st.session_state.data_set:
            col1.markdown('-----')
            col1.subheader(self.__STRINGS['Header2'])
            
            ########## Scaler Selector ##########
            scaler_entry = col1.radio(self.__STRINGS['Scaler_Entry'],
                                      options = self.__STRINGS['Scaler_Options'],
                                      index = 0,
                                      key = 'scaler_input',
                                      horizontal = True)

            ########## Classifier Selector ##########
            classifier = col1.selectbox(self.__STRINGS['Classifier_Entry'], 
                                        options = self.__STRINGS['Classifier_Options'], 
                                        key = 'classifier_input')
            
            subcol1_1, subcol2_1, subcol3_1, subcol4_1 = col1.columns(4)

            # columns for parameter inputs
            params = [subcol1_1.empty(), subcol2_1.empty(), subcol3_1.empty(), subcol4_1.empty()]
            validate_class = subcol1_1.empty()

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
                    
                    temp = params[i].text_input(label = param_name, value = value)
                    param_inputs[param_name] = temp

                st.session_state.param_dict[classifier] = param_inputs
                
                validate_class.button('Validate',
                                      key = 'val_class',
                                      on_click=self.__validate_class_clust,
                                      args = [st.session_state.classifier_input,
                                              'classifier'])
                
                validate_class_res = col1.empty()
                
                if st.session_state.class_results_err is not None:
                    validate_class_res.error(st.session_state.class_results_err)
                elif st.session_state.class_results is not None:
                    validate_class_res.write(f"Accuracy Score: {st.session_state.class_results}")
            
            ########## Cluster Selector ##########
            cluster = col1.selectbox(self.__STRINGS['Cluster_Entry'], 
                                        options = self.__STRINGS['Cluster_Options'], 
                                        key = 'cluster_input')
            
            subcol1_2, subcol2_2, subcol3_2, subcol4_2 = col1.columns(4)

            # columns for parameter inputs
            params = [subcol1_2.empty(), subcol2_2.empty(), subcol3_2.empty(), subcol4_2.empty()]
            validate_cluster = subcol1_2.empty()

            if cluster != self.__STRINGS['Default_Selectbox'][0]:
                param_inputs = {}
                # generate each parameter text input and collect their values
                for i in range(len(self.__STRINGS["Cluster_Params"][cluster])):
                    param_name = self.__STRINGS["Cluster_Params"][cluster][i]
                    # Set default value to previous entry, otherwise empty
                    try:
                        value = st.session_state.param_dict[cluster][param_name]
                    except:
                        value = None
                    
                    temp = params[i].text_input(param_name, value)
                    param_inputs[param_name] = temp

                st.session_state.param_dict[cluster] = param_inputs
                
                validate_cluster.button('Validate', 
                                        key = 'val_clus',
                                        on_click=self.__validate_class_clust,
                                        args=[st.session_state.cluster_input, 
                                              'cluster'])
                
                validate_clust_res = col1.empty()
                
                if st.session_state.clust_results_err is not None:
                    validate_clust_res.error(st.session_state.clust_results_err)
                elif st.session_state.clust_results is not None:
                    validate_clust_res.write(f"Accuracy Score: {st.session_state.clust_results}")
        # endregion
            if (st.session_state.classifier_input != self.__STRINGS['Default_Selectbox'][0]) and \
                (st.session_state.cluster_input != self.__STRINGS['Default_Selectbox'][0]):
                col1.markdown('-----')
                run_button = col1.button(self.__STRINGS['Run_Button'],
                                        key = 'run_button',
                                        on_click=self.__run)
            
            run_error = col1.empty()

            if st.session_state.run_err is not None:
                run_error.error(st.session_state.run_err)

        # Results
        # region

        col2.subheader(self.__STRINGS['Header3'])
        
        results_text = col2.empty()
        if st.session_state.run_results is not None:
            run_results = str(st.session_state.run_results).replace("'", '').replace('[', '').replace(']','')
            results_text.write(f"Potentially misclassified points: {run_results}")

        ########## Raw Data Toggle ##########
        raw_data = col2.toggle(self.__STRINGS['Show_Data_Toggle'],
                               key = 'raw_toggle',
                               disabled = not st.session_state.has_data)

        if raw_data:
            col2.dataframe(st.session_state.data)
            if st.session_state.data_set:
                col2.dataframe(st.session_state.redefine.get_X())
        # endregion
        
        ###### TEMP ######
        


        col2.write(st.session_state)


if __name__ == '__main__':
    app = App()
    app.window()