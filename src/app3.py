from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import json
import os

from bokeh.embed import file_html
from bokeh.resources import CDN

from redefine import REDEFINE

class App:
    
    with open('./src/strings.txt', 'r') as f:
        __STRINGS = json.load(f)

    __DEMO_DATA_PATH = './data/iris_modified.csv'

    def __init__(self):
        if 'ss_data' not in st.session_state:
            st.session_state.ss_data = None
        if 'ss_data_demo_b' not in st.session_state:
            st.session_state.ss_data_demo_b = False
        if 'ss_redefine' not in st.session_state:
            st.session_state.ss_redefine = None

        if 'ss_data_raw_b' not in st.session_state:
            st.session_state.ss_data_raw_b = False

        if 'ss_param_rand_b' not in st.session_state:
            st.session_state.ss_param_rand_b = False

        if 'ss_param_dict' not in st.session_state:
            st.session_state.ss_param_dict = {}

        if 'ss_param_super_model' not in st.session_state:
            st.session_state.ss_param_super_model = self.__STRINGS['Default_Selectbox'][0]
        if 'ss_param_unsup_model' not in st.session_state:
            st.session_state.ss_param_unsup_model = self.__STRINGS['Default_Selectbox'][0]

        if 'ss_param_kf_seed' not in st.session_state:
            st.session_state.ss_param_kf_seed = None

        if 'ss_results' not in st.session_state:
            st.session_state.ss_results = None
        return
    
    def __reset_state(self,
                      keys: str | list[str]):
        '''
        Sets session state variables to their neutral state, using the last part of the variable name as a flag for what their neutral state is.

        Parameters:
            keys: the name(s) of the session state variables that are to be reset.
        '''
        if type(keys) == str:
            keys = [keys]
        for k in keys:
            if k in st.session_state.keys():
                if k.split('_')[-1] == 'b':
                    st.session_state[k] = False
                elif k.split('_')[-1] == 'dict':
                    st.session_state[k] = {}
                else:
                    st.session_state[k] = None
        return
    
    def __load_demo_data(self):
        '''
        Loads the demo data as a pandas dataframe and adds it to the session state.
        '''

        data = pd.read_csv(self.__DEMO_DATA_PATH)
        st.session_state.ss_data = data
        return
    
    def __load_data(self, 
                    empty : st.empty):
        '''
        Reads in the uploaded data file and verifies it will work with the model.
        
        Parameters:
            empty: an empty streamlit container that displays any error found.
        '''

        @st.cache_data
        def verify_data(file : st.UploadedFile
                        ) -> pd.DataFrame | str:
            '''
            Reads the file as a .csv, then verifies if the data shape is valid.
            Caches this process to minimize processing time, especially in the face of mistakes.

            Parameters:
                file: a Streamlit UploadedFile object, from the file_uploader widget

            Returns:
                Either a Pandas DataFrame, if the data is valid, or an error string, if not.
            '''
            
            if file.name.split('.')[-1] != 'csv':
                return self.__STRINGS['Data_Entry_Error_Type']
            
            data = pd.read_csv(file)

            if len(data) == 0:
                return self.__STRINGS['Data_Entry_Error_File_Empty']

            data_map = data.map(np.isreal)
            num_numeric = sum([bool(data_map[x].all()) for x in data.columns])
            num_col = len(data.columns)
            if (num_col - num_numeric) > 2:
                return self.__STRINGS['Data_Entry_Error_Nonnumeric']
            
            return data
        
        data = verify_data(st.session_state.ss_data_file)
        if type(data) is str:
            empty.error(data)
            self.__reset_state(['ss_data_raw_b','ss_data'])
        else:
            st.session_state.ss_data = data

        return

    def __set_data(self, 
                   data_demo_b : bool, 
                   data : pd.DataFrame, 
                   data_file : st.UploadedFile, 
                   target_col : str, 
                   id_col : str, 
                   target_empty : st.empty, 
                   id_empty : st.empty):
        
        '''
        Verifies the columns and data then creates a REDEFINE object. Uses caching to minimize processing time.

        Parameters:
            data_demo_b: a boolean that indicates if the user wishes to use the demo data
            data: a Pandas DataFrame of the data
            data_file: the Streamlit UploadedFile object from the file_uploader widget
            target_col: the column name indicated as having the target values
            id_col: the column name indicated as having the ids for each observation, can be None
            target_empty: the streamlit container for if there is an error with the target column
            id_empty: the streamlit container for if there is an error with the id column or the data
        '''
        
        @st.cache_data
        def create_redefine_object(file_name, data, target_col, id_col):
            return REDEFINE(file_name, data.copy(), target_col, id_col)
        
        # If data and columns are valid:
        if data is not None:
            def_val = self.__STRINGS['Default_Selectbox'][0]
            if (target_col == def_val) or (id_col == def_val):
                if target_col == def_val:
                    
                    target_empty.error(self.__STRINGS['Column_Error_Empty'])
                if id_col == def_val:
                    id_empty.error(self.__STRINGS['Column_Error_Empty'])
                return
            elif target_col == id_col:
                id_empty.error(self.__STRINGS['Column_Error_Dup'])
                return
            else:
                try:
                    if data_demo_b:
                        file_name = os.path.basename(self.__DEMO_DATA_PATH)
                    else: 
                        file_name = data_file.name
                    st.session_state.ss_redefine = create_redefine_object(file_name,
                                                                          data,
                                                                          target_col,
                                                                          id_col)
                except Exception as e:
                    print(e)
                    id_empty.error(e)
                    return
                
    def __set_params(self, 
                     f : st.UploadedFile):
        '''
        Sets parameter variables according to the uploaded JSON file.

        Parameters:
            f: the uploaded parameter JSON file
        '''
        if 'super' in f:
            if 'scaler_name' in f['super']:
                st.session_state.ss_param_scaler = f['super']['scaler_name']

            if ('model_name' in f['super']) and ('model_params' in f['super']):
                st.session_state.ss_param_super_model = f['super']['model_name']
                st.session_state.ss_param_dict[st.session_state.ss_param_super_model] = f['super']['model_params']

        if ('unsup' in f) and ('model_name' in f['super']) and ('model_params' in f['super']):
            st.session_state.ss_param_unsup_model = f['unsup']['model_name']
            st.session_state.ss_param_dict[st.session_state.ss_param_unsup_model] = f['unsup']['model_params']

        if 'kf_random_seed' in f:
            st.session_state.ss_param_kf_seed = f['kf_random_seed']
        else:
            st.session_state.ss_param_kf_seed = None
            
        return
                
    def __validate_model(self, 
                         model_type : str, 
                         model_str : str, 
                         res_empty : st.empty):
        '''
        Calls the run_model function from redefine.py then displays the results.

        Parameters:
            model_type: a string of either 'supervised' or 'unsupervised' to trigger the correct type of model
            model_str: the name of the model chosen, as listed in the dropdown menu
            res_empty: the streamlit container that displays either the accuracy score or the error
        '''
        try:
            info = st.session_state.ss_redefine.run_model(model_str = model_str,
                                                          params = st.session_state.ss_param_dict[model_str],
                                                          scaler_str = st.session_state.ss_param_scaler,
                                                          model_type = model_type,
                                                          keep_rand = st.session_state.ss_param_rand_b)
            res_empty.text(f"Accuracy Score: {info['score']}")
        except Exception as e:
            res_empty.error(e)
        return
    
    def __run(self, 
              run_empty : st.empty):
        '''
        Runs the model. Saves the results in session state variables, displays errors if they occur.

        Parameters:
            run_empty: the streamlit container for displaying any errors that occur.
        '''
        print("Run!")
        self.__reset_state(['ss_results', 'ss_results_files', 'ss_results_plots'])
        supervised = st.session_state.ss_param_super_model
        unsupervised = st.session_state.ss_param_unsup_model
        try:
            results, files, plots = st.session_state.ss_redefine.run_redefine(supervised,
                                                                              st.session_state.ss_param_dict[supervised],
                                                                              unsupervised,
                                                                              st.session_state.ss_param_dict[unsupervised],
                                                                              st.session_state.ss_param_scaler,
                                                                              st.session_state.ss_param_rand_b,
                                                                              st.session_state.ss_param_kf_seed)
            st.session_state.ss_results = results
            st.session_state.ss_results_files = files
            st.session_state.ss_results_plots = plots

        except Exception as e:
            run_empty.error(e)

        return
    
    def __data_entry_ui(self, col1):
        '''
        The display for Step 1, the data entry and validation.
        '''
        
        data_demo_b = col1.toggle(self.__STRINGS['Data_Demo'],
                                  key = 'ss_data_demo_b',
                                  help = self.__STRINGS['Data_Demo_Help'],
                                  on_change = self.__reset_state,
                                  args=[['ss_data_raw_b','ss_data', 'ss_redefine', 'ss_param_json_b', 'ss_param_rand_b']])
        
        data_file = col1.file_uploader(self.__STRINGS['Data_Entry'], 
                                       type = 'csv', 
                                       accept_multiple_files = False,
                                       key = 'ss_data_file',
                                       help = self.__STRINGS['Data_Entry_Help'],
                                       on_change = self.__reset_state,
                                       args = [['ss_data_raw_b','ss_data', 'ss_redefine', 'ss_param_json_b', 'ss_param_rand_b']],
                                       disabled = data_demo_b)
        data_file_err = col1.empty()

        if data_demo_b:
            self.__load_demo_data()
        elif data_file:
            self.__load_data(data_file_err)
        else:
            data_file_err.error(self.__STRINGS['Data_Entry_Error_Empty'])
        
        ########## Raw Data Toggle ##########

        data_raw_b = col1.toggle(self.__STRINGS['Show_Data_Toggle'],
                                 key = 'ss_data_raw_b',
                                 disabled = (st.session_state.ss_data is None))
        if data_raw_b:
            col1.dataframe(st.session_state.ss_data)

        # Column Picking
        cols_full = self.__STRINGS['Default_Selectbox'].copy()
        # cols_full = col_default.copy()
        if st.session_state.ss_data is not None:
            cols_full.extend(st.session_state.ss_data.columns)

            if st.session_state.ss_data_demo_b:
                st.session_state.ss_data_target_col = self.__STRINGS['Data_Demo_Cols']['target']
                st.session_state.ss_data_id_col = self.__STRINGS['Data_Demo_Cols']['id']

        data_target_col = col1.selectbox(self.__STRINGS['Target_Entry'],
                                         options = cols_full.copy(),
                                         key = 'ss_data_target_col',
                                         help = self.__STRINGS['Target_Entry_Help'],
                                         disabled = data_demo_b)
        data_target_err = col1.empty()

        cols_full.insert(1, self.__STRINGS['None_Selectbox_Option'])

        data_id_col = col1.selectbox(self.__STRINGS['ID_Entry'],
                                     options = cols_full.copy(),
                                     key = 'ss_data_id_col',
                                     help = self.__STRINGS['ID_Entry_Help'],
                                     disabled = data_demo_b)
        data_id_err = col1.empty()

        ########## Set Data Button ##########
        data_set = col1.button(self.__STRINGS['Data_Set_Button'],
                                  key = 'ss_data_set_b')
        if data_set:
            self.__set_data(data_demo_b, 
                            st.session_state.ss_data,
                            data_file,
                            data_target_col,
                            data_id_col, 
                            data_target_err, 
                            data_id_err)
        return
    
    def __param_entry_ui(self, col1):
        '''
        The display for Step 2, the parameter entry and validation.
        '''

        col1_1, col1_2 = col1.columns(2)
        param_json_b = col1_1.toggle(label = self.__STRINGS['Param_Upload'],
                                     key = 'ss_param_json_b',
                                     help = self.__STRINGS['Param_Upload_Help'],
                                     )
        
        if param_json_b:
            param_rand_b = col1_2.checkbox(label = self.__STRINGS['Param_Rand_Check'],
                                           key = 'ss_param_rand_b',
                                           help = self.__STRINGS['Param_Rand_Help'])
            
            param_file = col1.file_uploader(label = self.__STRINGS['Param_File_Upload'],
                                            type = "json",
                                            key = 'ss_param_file')
            param_file_err = col1.empty()
        else:
            param_file = None
            
        param_reset_b = col1.button(label = self.__STRINGS['Param_Reset'],
                                    help = self.__STRINGS['Param_Reset_Help'])
        if param_reset_b:
            self.__reset_state('ss_param_dict')
            if param_file is not None:
                if param_file.name.split('_')[0] == 'params':
                    self.__set_params(json.load(param_file))
                else:
                    param_file_err.error(self.__STRINGS['Param_File_Error'])
                
        
        ########## Scaler Selector ##########
        param_scaler = col1.radio(self.__STRINGS['Scaler_Entry'],
                                    options = self.__STRINGS['Scaler_Options'],
                                    index = 0,
                                    help = self.__STRINGS['Scaler_Entry_Help'],
                                    key = 'ss_param_scaler',
                                    horizontal = True)

        ########## Supervised Model Selector ##########
        param_super_model = col1.selectbox(self.__STRINGS['Supervised_Entry'], 
                                            options = self.__STRINGS['Supervised_Options'], 
                                            key = 'ss_param_super_model',
                                            help = self.__STRINGS['Supervised_Entry_Help'])
        
        # Set up containers
        subcol1_1, subcol1_2, subcol1_3, subcol1_4 = col1.columns(4)
        param_super_conts = [subcol1_1.empty(), subcol1_2.empty(), subcol1_3.empty(), subcol1_4.empty()]

        if param_super_model != self.__STRINGS['Default_Selectbox'][0]:
            if param_super_model not in st.session_state.ss_param_dict:
                st.session_state.ss_param_dict[param_super_model] = {}
            param_super_inputs = st.session_state.ss_param_dict[param_super_model]
            
            # generate each parameter text input and collect values
            for i in range(len(self.__STRINGS['Supervised_Params'][param_super_model])):
                param_name = self.__STRINGS['Supervised_Params'][param_super_model][i]
                param_help = self.__STRINGS['Supervised_Params_Help'][param_super_model][i]

                # Set default value to previous entry, otherwise empty
                try:
                    default = st.session_state.ss_param_dict[param_super_model][param_name]
                except:
                    default = None
                
                value = param_super_conts[i].text_input(label = param_name,
                                                        value = default,
                                                        help = param_help)
                param_super_inputs[param_name] = value
            
            st.session_state.ss_param_dict[param_super_model] = param_super_inputs

            param_super_val_b = subcol1_1.button(self.__STRINGS['Validate_Button'],
                                            key = 'ss_param_super_val_b')
            
            param_super_res_cont = col1.empty()

            if param_super_val_b:
                self.__validate_model('supervised', param_super_model, param_super_res_cont)

        ########## Unsupervised Model Selector ##########
        param_unsup_model = col1.selectbox(self.__STRINGS['Unsupervised_Entry'], 
                                            options = self.__STRINGS['Unsupervised_Options'], 
                                            key = 'ss_param_unsup_model',
                                            help = self.__STRINGS['Unsupervised_Entry_Help'])
        
        # Set up containers
        subcol2_1, subcol2_2, subcol2_3, subcol2_4 = col1.columns(4)
        param_unsup_conts = [subcol2_1.empty(), subcol2_2.empty(), subcol2_3.empty(), subcol2_4.empty()]

        if param_unsup_model != self.__STRINGS['Default_Selectbox'][0]:
            if param_unsup_model not in st.session_state.ss_param_dict:
                st.session_state.ss_param_dict[param_unsup_model] = {}
            param_unsup_inputs = st.session_state.ss_param_dict[param_unsup_model]
            
            # generate each parameter text input and collect values
            for i in range(len(self.__STRINGS['Unsupervised_Params'][param_unsup_model])):
                param_name = self.__STRINGS['Unsupervised_Params'][param_unsup_model][i]
                param_help = self.__STRINGS['Unsupervised_Params_Help'][param_unsup_model][i]

                # Set default value to previous entry, otherwise empty
                try:
                    default = st.session_state.ss_param_dict[param_unsup_model][param_name]
                except:
                    default = None
                
                value = param_unsup_conts[i].text_input(label = param_name,
                                                        value = default,
                                                        help = param_help)
                param_unsup_inputs[param_name] = value
            
            st.session_state.ss_param_dict[param_unsup_model] = param_unsup_inputs

            param_unsup_val_b = subcol2_1.button(self.__STRINGS['Validate_Button'],
                                            key = 'ss_param_unsup_val_b')
            
            param_unsup_res_cont = col1.empty()

            if param_unsup_val_b:
                self.__validate_model('unsupervised', param_unsup_model, param_unsup_res_cont)
        
        return

    def __results_ui(self, col2):
        '''
        The display for the model results, on the right side of the screen.
        '''

        run_results = str(st.session_state.ss_results).replace("'", '').replace('[', '').replace(']','')
        col2.write(f"{self.__STRINGS['Results_Text']}\t{run_results}")

        show_plot = col2.toggle(self.__STRINGS['Show_Plot_Toggle'],
                               key = 'plot_toggle',
                               disabled = (st.session_state.ss_results_plots is None))

        if show_plot:
            col2.radio(self.__STRINGS['Plot_Type'], 
                       self.__STRINGS['Plot_Type_Options'],
                       index = 0,
                       horizontal = True,
                       key = 'ss_results_plot_type')

            pindex = self.__STRINGS['Plot_Type_Options'].index(st.session_state.ss_results_plot_type)
            col2.bokeh_chart(st.session_state.ss_results_plots[pindex], True)
            col2.download_button(label = self.__STRINGS['Download_Plot'], 
                                 data = file_html(st.session_state.ss_results_plots[pindex], CDN, 'plot'),
                                 file_name = 'plot.html')

        if st.session_state.ss_results_files is not None:
            results_path, metadata_path, params_path = st.session_state.ss_results_files
            
            result_f = open(results_path, 'r')
            metadata_f = open(metadata_path, 'r')
            params_f = open(params_path, 'r')
            col2.download_button(label = self.__STRINGS['Download_Results'],
                                 data = result_f,
                                 file_name = results_path.split('/')[-1])
            
            col2.download_button(label = self.__STRINGS['Download_Metadata'],
                                 data = metadata_f,
                                 file_name = metadata_path.split('/')[-1])
            
            col2.download_button(label = self.__STRINGS['Download_Parameters'],
                                 data = params_f,
                                 file_name = params_path.split('/')[-1])

            result_f.close()
            metadata_f.close()
            params_f.close()

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
        
        self.__data_entry_ui(col1)

        if st.session_state.ss_redefine is not None:
            col1.markdown('-----')
            col1.subheader(self.__STRINGS['Header2'])
            
            self.__param_entry_ui(col1)
            
        col1.markdown('-----')

        # Run Button
        if (st.session_state.ss_param_super_model != self.__STRINGS['Default_Selectbox'][0]) and \
           (st.session_state.ss_param_unsup_model != self.__STRINGS['Default_Selectbox'][0]):
            run_b = col1.button(self.__STRINGS['Run_Button'])
            run_cont = col1.empty()
            
            if run_b:
                self.__run(run_cont)

        col2.subheader(self.__STRINGS['Header3'])
        
        # Results
        
        if st.session_state.ss_results is not None:
            self.__results_ui(col2)
        return
    
    