import numpy as np
import pandas as pd
import os
import json
from datetime import datetime as dt
from traceback import print_exc as pe

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from bokeh.models import ColumnDataSource, Legend
from bokeh.plotting import figure
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Accent8

from typing import Union, Optional

class REDEFINE:
    def __init__(self, 
                 file_name : str, 
                 data : pd.DataFrame, 
                 target_col : str, 
                 id_col : str):
        '''
        Cleans and initializes data, and initializes constants

        Parameters:
            file_name: name of the data file
            data: dataframe of data
            target_col: name of the target column
            id_col: name of the id column, or '(None)' if no such column exists
        '''

        # set all data variables
        self.__clean_data(data, target_col, id_col)
        self.__file_name = file_name.split('.')[0]

        # Constants
        self.__kfolds = 10
        self.__PATH_OUT = './output/'

        self.__SCALERS = {"Standard" : StandardScaler,
                          "Min-Max" : MinMaxScaler,
                          "Absolute Max" : MaxAbsScaler,
                          "Robust" : RobustScaler}
        self.__MODELS = {"K Nearest Neighbors" : KNeighborsClassifier,
                         "Random Forest" : RandomForestClassifier,
                         "KMeans" : KMeans}
        
        return
    
    def __get_file_paths(self):
        '''
        File Naming Convention: [file_type]_[data_file_name]_[timestamp]

        Returns:
            results_path: the path and name of the results file
            metadata_path: the path and name of the metadata file
        '''
        time = self.__timestamp()
        results_path = os.path.join(self.__PATH_OUT, f"results_{self.__file_name}_{time}.csv")
        metadata_path = os.path.join(self.__PATH_OUT, f"metadata_{self.__file_name}_{time}.txt")
        params_path = os.path.join(self.__PATH_OUT, f"params_{self.__file_name}_{time}.json")

        return results_path, metadata_path, params_path
    
    def run_redefine(self,
                     super_str : str,
                     super_params : dict[str, str],
                     unsup_str : str,
                     unsup_params : dict,
                     scaler_str : str
                     ) -> tuple[Optional[str], Optional[list[Union[int, str]]], Optional[tuple[str]], Optional[tuple[figure]]]:
        '''
        The main pipeline.  Compares the results from the supervised model and unsupervised model to the original labels.  Saves results and metadata to files and outputs them to the app.

        Parameters:
            super_str: the name of the supervised model
            super_params: the parameters for the supervised model
            unsup_str: the name of the unsupervised model
            unsup_params: the parameters for the supervised model
            scaler_str: the name of the scaling model

        Returns:
            err: any error that occurs, returns everything else as None, returns None if no error
            results: list of IDs that got flagged as misclassified
            files: tuple of file paths for results and metadata, respectively
            plots: tuple of PCA and t-SNE Bokeh plots, respectively
        '''
        results_df = pd.DataFrame(columns=['Label','SupervisedResult', 'UnsupervisedResult', 'Flagged'], index=self.__IDs)
        results_df['Label'] = self.__Y
        try:
            # Run Supervised
            super_info = self.run_model(model_str = super_str,
                                        params = super_params, 
                                        scaler_str = scaler_str,
                                        model_type = "supervised",
                                        store_results = True)
            
            if super_info['error'] is not None:
                return super_info['error'], None, None, None
            else:
                for idx, res in super_info['results']:
                    results_df.at[idx, 'SupervisedResult'] = res
            
            # Run Cluster Alg
            unsup_info = self.run_model(model_str = unsup_str,
                                        params = unsup_params, 
                                        scaler_str = scaler_str,
                                        model_type = "unsupervised",
                                        store_results = True)
            
            if unsup_info['error'] is not None:
                return unsup_info['error'], None, None, None
            else:
                results_df['UnsupervisedResult'] = unsup_info['results']

            # Results
            flagged_ids = self.__eval_misclassed(results_df)

            path_names = self.__get_file_paths()
            
            plots, plot_random = self.__make_plots(flagged_ids, results_df, scaler_str)
        
            self.__write_to_files(results_df, super_info, unsup_info, flagged_ids, plot_random, path_names)

            return None, flagged_ids, path_names, plots
        except Exception as e:
            return e, None, None, None
        
    def run_model(self,
                  model_str : str, 
                  params : dict, 
                  scaler_str : str,
                  model_type : str,
                  store_results : bool = False
                  ) -> dict:
        '''
        Runs either K-Fold Cross-Validation on a supervised model or clustering with an unsupervised model.

        Parameters:
            model_str: name of the model as seen in the app; mapped to the model in self.__MODELS
            params: dictionary of model parameters
            scaler_str: name of the scaling model as seen in the app; mapped to the model in self.__SCALERS
            model_type: 'supervised' or 'unsupervised'
            store_results: False for validation, True for running results
        
        Returns:
            info: dictionary with metadata: model_name, model_params, scaler_name, error, score, results, runtime
        '''

        info = {'model_name' : model_str, 
                'scaler_name' : scaler_str, 
                'error' : None, 
                'score' : None,
                'results' : None}

        clean_params = self.__clean_params(model_str, params)
        info['model_params'] = clean_params
        
        t0 = dt.now()
        try:
            model = self.__MODELS[model_str](**clean_params)
            if scaler_str == "None":
                scaler = None
            else:
                scaler = self.__SCALERS[scaler_str]()

            if model_type == 'supervised':
                score, results = self.__doKFold(model, scaler, store_results)
            else:
                score, results = self.__doClustering(model, scaler)
            
            info['score'] = score
            info['results'] = results

        except Exception as e:
            info['error'] = str(e)
        info['runtime'] = dt.now() - t0
        return info
    
    def __clean_data(self, 
                     data : pd.DataFrame, 
                     target_col : str, 
                     id_col : str):
        '''
        Cleans data and initializes class variables.
        
        Parameters:
            data: dataframe of data
            target_col: name of the target column
            id_col: name of the id column, or '(None)' if no such column exists
        '''

        # Create index column if needed
        if id_col == "(None)":
            data = data.reset_index()
            id_col = "index"
        
        self.__IDs = data[id_col]

        # get classes
        Y = data[target_col]
        self.__Y_names = Y.unique()
        self.__Y = Y.values

        # remove target and ID columns from data
        data_cols = list(data.columns).copy()
        data_cols.remove(target_col)
        data_cols.remove(id_col)
        self.__X = data[data_cols].values
        return
    
    def __clean_params(self, 
                       model_str : str, 
                       params : dict
                       ) -> dict:
        '''
        Cleans parameters and initializes random states.
        
        Parameters:
            model_str: name of the model, as seen in the app; mapped to model in self.__MODELS
            params: dictionary of parameters

        Returns:
            clean_param: cleaned dictionary of parameters
        '''
        
        # Remove empty parameters and convert strings to numbers where necessary
        clean_param = { key:self.__str_to_num(val) 
                       for (key, val) in params.items() 
                       if (val != "") and (val is not None) }
        
        # get all possible model parameters
        model_params = self.__MODELS[model_str]().get_params().keys()

        # See if random_state is a parameter, set seed for replicability
        if 'random_state' in model_params:
            # TODO: set boolean to keep/change random seed
            random_seed = self.__get_random_seed()
            clean_param['random_state'] = random_seed

        # See if n_clusters is a parameter, set to len of Y_names
        if 'n_clusters' in model_params:
            clean_param['n_clusters'] = len(self.__Y_names)
        
        # set n_init for KMeans
        if 'n_init' in model_params:
            clean_param['n_init'] = 20

        return clean_param

    def __eval_misclassed(self, 
                          results_df : pd.DataFrame
                          ) -> list:
        '''
        Label and return the points where supervised and unsupervised results agree on a class that is different than the original label.

        Parameters: 
            results_df: the dataframe with the IDs (as the index), original labels, supervised results and unsupervised results; gets modified, adds True value in the 'Flagged' column the point is marked as p
        
        Returns:
            list of IDs for the points that got flagged as misclassified
        '''
        
        flagged_idx = []
        for idx, row in results_df.iterrows():
            if (row['SupervisedResult'] == row['UnsupervisedResult']) and \
            (row['Label'] != row['SupervisedResult']):
                flagged_idx.append(idx)
                results_df.at[idx, 'Flagged'] = True
        return flagged_idx
    
    def __write_to_files(self, 
                         results_df : pd.DataFrame, 
                         super_info : dict, 
                         unsup_info : dict, 
                         flagged_ids : list, 
                         plot_random : int, 
                         path_names : tuple):
        '''
        Write the results and metadata files to the given file paths.

        Parameters:
            results_df: the DataFrame that holds the results from the supervised and unsupervised models
            super_info: dictionary with all of the recorded data about the supervised model
            unsup_info: dictionary with all of the recorded data about the unsupervised model
            flagged_ids: list of IDs for the flagged points
            plot_random: the random seed provided to both PCA and t-SNE for creating the plot
            results_path: path and name of the results file
            metadata_path: path and name of the metadata file
        '''
        results_path, metadata_path, params_path = path_names
        # Results file
        results_df.to_csv(results_path)

        # Metadata file
        with open(metadata_path, 'w') as f:
            f.write('Metadata\n\n')

            f.write(f"Flagged Points: {flagged_ids}\n\n")

            f.write(f"KFold Random Seed: {self.__kf_random_seed}\n\n")
            f.write("Supervised Model:\n")
            for (key, val) in super_info.items():
                if key != 'results':
                    if key == 'model_name':
                        val = str(self.__MODELS[val]).split("'")[1]
                    if (key == 'scaler_name') and (val != 'None'):
                        val = str(self.__SCALERS[val]).split("'")[1]
                    f.write(f"{key}: {val}\n")
            f.write("\n")
            f.write("Unsupervised Model:\n")
            for (key, val) in unsup_info.items():
                if key != 'results':
                    if key == 'model_name':
                        val = str(self.__MODELS[val]).split("'")[1]
                    if (key == 'scaler_name') and (val != 'None'):
                        val = str(self.__SCALERS[val]).split("'")[1]
                    f.write(f"{key}: {val}\n")

            f.write(f"Plot Random Seed: {plot_random}")

        # Params file

        keep = ['model_name', 'scaler_name', 'model_params']

        super_params = {key:val for key, val in super_info.items() if key in keep}
        unsup_params = {key:val for key, val in unsup_info.items() if key in keep}

        params_dict = {'super': super_params, 'unsup': unsup_params}
        with open(params_path, 'w+') as f:
            json.dump(params_dict, f)

        return

    def __make_plots(self, 
                     flagged_ids : list, 
                     results_df : pd.DataFrame, 
                     scaler_str : str
                     ) -> tuple[tuple[figure], int]:
        '''
        Scale the data and set the random seed for PCA and t-SNE, then generate the plots.

        Parameters:
            flagged_ids: list of IDs for the flagged points
            results_df: the DataFrame that holds the results from the supervised and unsupervised models
            scaler_str: name of the scaling model as seen in the app; mapped to the model in self.__SCALERS
        
        Returns:
            plots: tuple of PCA and t-SNE Bokeh plots, respectively
            random_state: the random state used for both PCA and t-SNE
        '''

        random_state = self.__get_random_seed()

        if scaler_str == "None":
            scaler = None
            x = self.__X.copy()
        else:
            scaler = self.__SCALERS[scaler_str]()
            x = scaler.fit_transform(self.__X)

        pca_plot = self.__make_plot(flagged_ids, results_df, x, 'pca', random_state)
        tsne_plot = self.__make_plot(flagged_ids, results_df, x, 'tsne', random_state)

        return (pca_plot, tsne_plot), random_state

    def __make_plot(self, 
                    flagged_ids : list, 
                    results_df : pd.DataFrame, 
                    x : np.ndarray,
                    plot_type : str, 
                    random_state : int
                    ) -> figure:
        '''
        Generate the bokeh figure object for the given data.

        Parameters:
            flagged_ids: list of IDs for the flagged points
            results_df: the DataFrame that holds the results from the supervised and unsupervised models
            x: the scaled data (if chosen scaler was not none)
            plot_type: type of plot, either 'pca' or 'tsne'
            random_state: the random state used for the dimensionality reduction method
        
        Returns:
            figure: bokeh figure with all points and flagged points in red
        '''

        if plot_type == 'pca':
            pca = PCA(n_components=2, random_state=random_state)
            x = pca.fit_transform(x)
        else:
            tsne = TSNE(n_components=2, random_state=random_state)
            x = tsne.fit_transform(x)


        TOOLTIPS = [
            ("ID", "@id"),
            ("Original Label", "@label"),
            ("Supervised Label", "@sup_label"),
            ("Unsupervised Label", "@unsup_label")
        ]

        p = figure(width=400, height=500,
                   tooltips = TOOLTIPS)

        p.title.text = plot_type.upper()
        p.axis.visible = False
        
        flag_idxs = np.nonzero(np.array(flagged_ids)[:, None] == np.array(self.__IDs))[1]

        x_true = np.delete(x, flag_idxs, axis=0)
        results_true = results_df.copy().drop(flagged_ids, axis=0)
        
        full_data = ColumnDataSource(dict(
            x1=x_true[:,0],
            x2=x_true[:,1],
            label=list(map(str, results_true['Label'])),
            sup_label=list(map(str, results_true['SupervisedResult'])),
            unsup_label=list(map(str, results_true['UnsupervisedResult'])),
            id = results_true.index
        ))
        
        color_mapper = CategoricalColorMapper(factors=list(map(str, self.__Y_names)), palette=Accent8)

        true_points = p.circle(x='x1', y='x2', source=full_data, size=7, alpha=0.8,
                            color={'field': 'label', 'transform': color_mapper},
                            legend_field='label')

        # flagged points
        flagged_X = x[flag_idxs]
        flagged_results = results_df[results_df.index.isin(flagged_ids)]
        
        flagged_data = ColumnDataSource(dict(
            x1=flagged_X[:,0],
            x2=flagged_X[:,1],
            label=list(map(str, flagged_results['Label'])),
            sup_label=list(map(str, flagged_results['SupervisedResult'])),
            unsup_label=list(map(str, flagged_results['UnsupervisedResult'])),
            id = flagged_results.index
        ))

        misclass_points = p.circle(x='x1', y='x2', source=flagged_data, size=7, alpha=0.8,
                                color='red', legend_label='Potentially Misclassified Points')

        p.legend.visible = False

        leg = Legend(items=p.legend.items, location='left')
        
        p.add_layout(leg, 'below')

        return p
    
    def __doKFold(self, 
                  model, 
                  scaler,
                  store_results : bool
                  ) -> tuple[float, zip]:
        '''
        Performs K-Fold cross-validation for supervised learning.

        Parameters:
            model: the supervised learning model 
            scaler: the scaler model
            store_results: True if running the full pipeline, False if running only the validation
        
        Returns:
            mean accuracy score: the mean accuracy score after the given number of K-Folds
            results zip: a zip of the IDs and their corresponding labels
        '''

        n = len(self.__X)
        idxs = np.linspace(0, n, self.__kfolds+1).astype(int)
        idx = np.arange(0, n)

        self.__kf_random_seed = self.__get_random_seed()
        rs = np.random.RandomState(self.__kf_random_seed)
        rs.shuffle(idx)

        x = self.__X.copy()[idx]
        y = self.__Y.copy()[idx]
        ids = self.__IDs.copy()[idx]

        accuracy_scores = np.zeros(self.__kfolds)

        id_res = np.zeros(n, dtype='O')
        yhat_res = np.zeros(n, dtype='O')

        for k, i in enumerate(range(1, len(idxs))):
            idx1 = idxs[i-1]
            idx2 = idxs[i]

            idtest = ids[idx1:idx2]

            xtest = x[idx1:idx2]
            xtrain = np.concatenate([x[idxs[0]:idx1], x[idx2:idxs[-1]]])

            ytest = y[idx1:idx2]
            ytrain = np.concatenate([y[idxs[0]:idx1], y[idx2:idxs[-1]]])

            if scaler is not None:
                xtrain = scaler.fit_transform(xtrain)
                xtest = scaler.transform(xtest)

            model.fit(xtrain, ytrain)
            yhat = model.predict(xtest)
            
            accuracy_scores[k] = self.__accuracy_score(ytest, yhat)

            if store_results:
                id_res[idx1:idx2] = ids[idx1:idx2]
                yhat_res[idx1:idx2] = yhat

        return np.mean(accuracy_scores), zip(id_res, yhat_res)

    def __doClustering(self,  
                       model, 
                       scaler
                       ) -> tuple[int, np.ndarray]:
        '''
        Performs clustering with an unsupervised learning model, then assigns categorical labels to each cluster.

        Parameters: 
            model: the unsupervised learning model
            scaler: the scaler model
        
        Returns:
            accuracy score: the accuracy score comparing the labeled clusters to the original labels
            yhat: the predicted labels of each point
        '''
        
        x = self.__X.copy()
        y = self.__Y.copy()
        ids = self.__IDs.copy()
        
        if scaler is not None:
            x = scaler.fit_transform(self.__X)
        
        yhat = model.fit_predict(x)

        # Relabel cluster names
        label_map = {}

        for i in range(len(self.__Y_names)):
            where_i = np.where(y == self.__Y_names[i])
            val_i = np.bincount(yhat[where_i]).argmax()
            label_map[val_i] = self.__Y_names[i]

        if len(label_map) == len(self.__Y_names):
            yhat = np.vectorize(label_map.__getitem__)(yhat)
        else:
            raise Exception("Model did not find the expected number of clusters.")
        
        return self.__accuracy_score(y, yhat), yhat
        
    # Misc
    def __str_to_num(self, 
                     n : str
                     ) -> Union[int, float,  str]:
        '''
        Converts a numeric string to a numeric type.
        
        Parameters:
            n: a potentially numeric string
        
        '''
        if n.isnumeric():
            return int(n)
        else:
            try:
                return float(n)
            except:
                return n
    
    def __get_random_seed(self) -> int:
        '''
        Returns:
            int: a randomly generated number
        '''
        return np.random.randint(0,100000000)
    
    def __accuracy_score(self, 
                         ytest : np.ndarray, 
                         yhat : np.ndarray
                         ) -> float:
        '''
        Calculates the accuracy score between two lists. Divides the number of correctly labeled data by the total number of data. Usually used to calculate the 'correctness' of a classification model.

        Parameters:
            ytest: the true data labels
            yhat: the predicted data labels
        
        Returns:
            float: the accuracy score of ytest and yhat
        '''
        nz = np.flatnonzero(ytest == yhat)
        return len(nz)/len(ytest)

    def __timestamp(self) -> str:
        '''
        Returns:
            str: current time with underscores for file-naming
        '''
        return str(dt.now()).replace(':','-').replace(' ', '_').replace('.', '_')

    # Getters
    def get_X(self):
        return self.__X.copy()
    
    def get_Y(self):
        return self.__Y.copy()
    
    def get_Y_names(self):
        return self.__Y_names.copy()
    
    def get_IDs(self):
        return self.__IDs.copy()
