from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import json
import os

from bokeh.embed import file_html
from bokeh.resources import CDN

from redefine import REDEFINE

class App:

    with open('main/strings.txt', 'r') as f:
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
        
        if 'super_results_err' not in st.session_state:
            st.session_state.super_results_err = None
        if 'super_results' not in st.session_state:
            st.session_state.super_results = None
        
        if 'unsup_results_err' not in st.session_state:
            st.session_state.unsup_results_err = None
        if 'unsup_results' not in st.session_state:
            st.session_state.unsup_results = None

        if 'run_err' not in st.session_state:
            st.session_state.run_err = None
        if 'run_results' not in st.session_state:
            st.session_state.run_results = None
        if 'run_files' not in st.session_state:
            st.session_state.run_files = None
        if 'run_plots' not in st.session_state:
            st.session_state.run_plots = None

        if 'raw_toggle' not in st.session_state:
            st.session_state.raw_toggle = False

        return
    
    def __to_state_false(self,
                         key: str | list[str]):
        '''
        Turns the given session state(s) false.  
        Used in on_click to reset states or turn functionalities off
        
        Parameters:
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
        
        Parameters:
            file: file via st.file_uploader | str if using demo data
        
        Returns:
            has_data: boolean for has_data flag
            data: pandas DataFrame of data if verified
            error: error string, if found
        '''

        @st.cache_data
        def verify_data(file : st.UploadedFile | str) -> (bool, pd.DataFrame, str):
            has_data = False
            data = ""
            error = ""

            if type(file) != str:
                if file is None:
                    error = self.__STRINGS['Data_Entry_Error_Empty']
                elif file.name.split('.')[-1] != 'csv':
                    error = self.__STRINGS['Data_Entry_Error_Type']

            if error == "":
                data = pd.read_csv(file)

                if len(data) == 0:
                    data = ""
                    error = self.__STRINGS['Data_Entry_Error_File_Empty']

                data_map = data.map(np.isreal)
                num_numeric = sum([bool(data_map[x].all()) for x in data.columns])
                num_col = len(data.columns)
                
                if (num_col - num_numeric) > 2:
                    data = ""
                    error = self.__STRINGS['Data_Entry_Error_Nonnumeric']
                
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
                file_name = st.session_state.data_input.name

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
    
    def __validate_model(self, 
                         model : str, 
                         model_type : str):
        '''
        Runs either a 10-fold cross validation with the chosen supervsied model and hyperparameters,
        or evaluates the unsupervised model and hyperparameters
        
        Parameters:
            model: model
            model_type: string, either 'supervised' or 'unsupervised'
        '''
        info = st.session_state.redefine.run_model(model, 
                                                   st.session_state.param_dict[model],
                                                   st.session_state.scaler_input,
                                                   model_type)
        if model_type == 'supervised':
            st.session_state.super_results_err = info['error']
            st.session_state.super_results = info['score']
        elif model_type == 'unsupervised':
            st.session_state.unsup_results_err = info['error']
            st.session_state.unsup_results = info['score']
        return
    
    def __run(self):
        print("Run!")
        supervised = st.session_state.super_input
        unsupervised = st.session_state.unsup_input
        err, results, files, plots = st.session_state.redefine.run_redefine(supervised,
                                                        st.session_state.param_dict[supervised],
                                                        unsupervised,
                                                        st.session_state.param_dict[unsupervised],
                                                        st.session_state.scaler_input)
        
        st.session_state.run_err = err
        st.session_state.run_results = results
        st.session_state.run_files = files
        st.session_state.run_plots = plots
        return
        
    def window(self):
        '''
        Main body for the UI.
        '''

        # Titles
        st.title(self.__STRINGS['Title'])
        st.write(self.__STRINGS['Description'])
        
        col1, col2 = st.columns([1,1])

        col1.subheader(self.__STRINGS['Header1'])
        # Data Input
        # region
        ########## Demo Data ##########
        demo_data = col1.toggle(self.__STRINGS['Demo_Data'],
                               key = 'demo_toggle',
                               help=self.__STRINGS['Demo_Data_Help'],
                               on_change = self.__to_state_false,
                               args = [['raw_toggle', 'data_set', 'plot_toggle']])

        ########## Data ##########
        data_file = col1.file_uploader(self.__STRINGS['Data_Entry'], 
                                       type = 'csv', 
                                       accept_multiple_files = False,
                                       key = 'data_input',
                                       help = self.__STRINGS['Data_Entry_Help'],
                                       on_change=self.__to_state_false,
                                       args=[['raw_toggle', 'plot_toggle', 'data_set']],
                                       disabled = st.session_state.demo_toggle)

        if demo_data:
            has_data, data, data_err_msg = self.__load_data(self.__DEMO_DATA_PATH)
            st.session_state.target_input = self.__STRINGS['Demo_Data_Cols']['target']
            st.session_state.id_input = self.__STRINGS['Demo_Data_Cols']['id']
        else:
            has_data, data, data_err_msg = self.__load_data(data_file)
        
        st.session_state.has_data = has_data
        st.session_state.data = data
        
        # Data error
        if data_err_msg != "":
            col1.error(data_err_msg)

        # Retrieve column names for selectboxes
        cols = self.__STRINGS['Default_Selectbox']
        cols_full = cols.copy()
        if st.session_state.has_data:
            cols_full.extend(list(data.columns))

        ########## Target column ##########
        target_col = col1.selectbox(self.__STRINGS['Target_Entry'],
                                    options = cols_full.copy(),
                                    key = 'target_input',
                                    help = self.__STRINGS['Target_Entry_Help'],
                                    on_change = self.__to_state_false,
                                    args = [['target_valid', 'plot_toggle', 'data_set']],
                                    disabled = st.session_state.demo_toggle)
        if st.session_state.has_data:
            if target_col == self.__STRINGS['Default_Selectbox'][0]:
                col1.error(self.__STRINGS['Column_Error_Empty'])
            else:
                st.session_state.target_valid = True

        ########## ID column and error message ##########
        cols_full.insert(1, self.__STRINGS['None_Selectbox_Option'])
        id_col = col1.selectbox(self.__STRINGS['ID_Entry'],
                                options = cols_full.copy(),
                                key = 'id_input',
                                help = self.__STRINGS['ID_Entry_Help'],
                                on_change = self.__to_state_false,
                                args = [['id_valid', 'plot_toggle', 'data_set']],
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
        # Only show Supervised/Unsupervised options if data and columns are valid
        if st.session_state.data_set:
            col1.markdown('-----')
            col1.subheader(self.__STRINGS['Header2'])
            
            ########## Scaler Selector ##########
            scaler_entry = col1.radio(self.__STRINGS['Scaler_Entry'],
                                      options = self.__STRINGS['Scaler_Options'],
                                      index = 0,
                                      help = self.__STRINGS['Scaler_Entry_Help'],
                                      key = 'scaler_input',
                                      horizontal = True)

            ########## Supervised Model Selector ##########
            super_model = col1.selectbox(self.__STRINGS['Supervised_Entry'], 
                                        options = self.__STRINGS['Supervised_Options'], 
                                        key = 'super_input',
                                        help = self.__STRINGS['Supervised_Entry_Help'])
            
            subcol1_1, subcol2_1, subcol3_1, subcol4_1 = col1.columns(4)

            # columns for parameter inputs
            params = [subcol1_1.empty(), subcol2_1.empty(), subcol3_1.empty(), subcol4_1.empty()]
            val_super_cont = subcol1_1.empty()

            if super_model != self.__STRINGS['Default_Selectbox'][0]:
                param_inputs = {}
                # generate each parameter text input and collect their values
                for i in range(len(self.__STRINGS["Supervised_Params"][super_model])):
                    param_name = self.__STRINGS["Supervised_Params"][super_model][i]
                    param_help = self.__STRINGS["Supervised_Params_Help"][super_model][i]
                    # Set default value to previous entry, otherwise empty
                    try:
                        value = st.session_state.param_dict[super_model][param_name]
                    except:
                        value = None
                    
                    temp = params[i].text_input(label = param_name, value = value, help = param_help)
                    param_inputs[param_name] = temp

                st.session_state.param_dict[super_model] = param_inputs
                
                val_super_cont.button('Validate',
                                      key = 'val_super',
                                      on_click=self.__validate_model,
                                      args = [st.session_state.super_input,
                                              'supervised'])
                
                val_super_res_cont = col1.empty()
                
                if st.session_state.super_results_err is not None:
                    val_super_res_cont.error(st.session_state.super_results_err)
                elif st.session_state.val_super:
                    val_super_res_cont.write(f"Accuracy Score: {st.session_state.super_results}")
            
            ########## Unsupervised Selector ##########
            unsup_model = col1.selectbox(self.__STRINGS['Unsupervised_Entry'], 
                                        options = self.__STRINGS['Unsupervised_Options'], 
                                        key = 'unsup_input',
                                        help = self.__STRINGS['Unsupervised_Entry_Help'])
            
            subcol1_2, subcol2_2, subcol3_2, subcol4_2 = col1.columns(4)

            # columns for parameter inputs
            params = [subcol1_2.empty(), subcol2_2.empty(), subcol3_2.empty(), subcol4_2.empty()]
            val_unsup_cont = subcol1_2.empty()

            if unsup_model != self.__STRINGS['Default_Selectbox'][0]:
                param_inputs = {}
                # generate each parameter text input and collect their values
                for i in range(len(self.__STRINGS["Unsupervised_Params"][unsup_model])):
                    param_name = self.__STRINGS["Unsupervised_Params"][unsup_model][i]
                    param_help = self.__STRINGS["Unsupervised_Params_Help"][unsup_model][i]
                    # Set default value to previous entry, otherwise empty
                    try:
                        value = st.session_state.param_dict[unsup_model][param_name]
                    except:
                        value = None
                    
                    temp = params[i].text_input(label = param_name, value = value, help = param_help)
                    param_inputs[param_name] = temp

                st.session_state.param_dict[unsup_model] = param_inputs
                
                val_unsup_cont.button('Validate', 
                                      key = 'val_unsup',
                                      on_click=self.__validate_model,
                                      args=[st.session_state.unsup_input, 
                                            'unsupervised'])
                
                val_unsup_res_cont = col1.empty()
                
                if st.session_state.unsup_results_err is not None:
                    val_unsup_res_cont.error(st.session_state.unsup_results_err)
                elif st.session_state.val_unsup:
                    val_unsup_res_cont.write(f"Accuracy Score: {st.session_state.unsup_results}")
        
            if (super_model != self.__STRINGS['Default_Selectbox'][0]) and \
                (unsup_model != self.__STRINGS['Default_Selectbox'][0]):
                col1.markdown('-----')
                run_button = col1.button(self.__STRINGS['Run_Button'],
                                        key = 'run_button',
                                        on_click=self.__run)
            
            run_error = col1.empty()

            if st.session_state.run_err is not None:
                run_error.error(st.session_state.run_err)
        # endregion

        # Results
        # region

        col2.subheader(self.__STRINGS['Header3'])
        
        results_text = col2.empty()
        if st.session_state.run_results is not None:
            run_results = str(st.session_state.run_results).replace("'", '').replace('[', '').replace(']','')
            results_text.write(f"{self.__STRINGS['Results_Text']}\t{run_results}")

        results_file1 = col2.empty()
        results_file2 = col2.empty()
        results_file3 = col2.empty()

        if st.session_state.run_files is not None:
            results_path, metadata_path, params_path = st.session_state.run_files

            result_f = open(results_path, 'r')
            metadata_f = open(metadata_path, 'r')
            params_f = open(params_path, 'r')

            results_file1.download_button(label = self.__STRINGS['Download_Results'],
                                          data = result_f,
                                          file_name = results_path)
            
            results_file2.download_button(label = self.__STRINGS['Download_Metadata'],
                                          data = metadata_f,
                                          file_name = metadata_path)
            
            results_file3.download_button(label = self.__STRINGS['Download_Parameters'],
                                          data = params_f,
                                          file_name = params_path)
            
            result_f.close()
            metadata_f.close()
            params_f.close()

        show_plot = col2.toggle(self.__STRINGS['Show_Plot_Toggle'],
                               key = 'plot_toggle',
                               disabled = (st.session_state.run_plots is None))

        if show_plot:
            col2.radio(self.__STRINGS['Plot_Type'], 
                       self.__STRINGS['Plot_Type_Options'],
                       index = 0,
                       horizontal = True,
                       key = 'plot_type')

            pindex = self.__STRINGS['Plot_Type_Options'].index(st.session_state.plot_type)
            col2.bokeh_chart(st.session_state.run_plots[pindex], True)
            col2.download_button(label = self.__STRINGS['Download_Plot'], 
                                 data = file_html(st.session_state.run_plots[pindex], CDN, 'plot'),
                                 file_name = 'plot.html')

        ########## Raw Data Toggle ##########
        raw_data = col2.toggle(self.__STRINGS['Show_Data_Toggle'],
                               key = 'raw_toggle',
                               disabled = not st.session_state.has_data)

        if raw_data:
            col2.dataframe(st.session_state.data)
        #endregion