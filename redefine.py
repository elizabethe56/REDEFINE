import numpy as np
import pandas as pd
import os
from datetime import datetime as dt

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

class REDEFINE:
    def __init__(self, file_name, data, target_col, id_col):
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
        self.__MODELS = {"Nearest Neighbor" : KNeighborsClassifier,
                         "Random Forest" : RandomForestClassifier,
                         "KMeans" : KMeans}
        
        return
    
    def __clean_data(self, data, target_col, id_col):

        if id_col == "(None)":
            data = data.reset_index()
            id_col = "index"

        self.__IDs = data[id_col]
        Y = data[target_col]
        self.__Y_names = Y.unique()
        self.__Y = Y.values

        data_cols = list(data.columns).copy()
        data_cols.remove(target_col)
        data_cols.remove(id_col)
        self.__X = data[data_cols].values
        return
    
    def __clean_params(self, model_str, params):
        # Remove empty parameters and convert strings to numbers where necessary
        clean_param = { key:self.__str_to_num(val) 
                       for (key, val) in params.items() 
                       if (val != "") and (val is not None) }
        # get all possible model parameters
        model_params = self.__MODELS[model_str]().get_params().keys()

        # See if random_state is a parameter, set seed for replicability
        if 'random_state' in model_params:
            random_seed = self.__get_random_seed()
            clean_param['random_state'] = random_seed

        # See if n_clusters is a parameter, set to len of Y_names
        if 'n_clusters' in model_params:
            clean_param['n_clusters'] = len(self.__Y_names)
        
        # set n_init for KMeans
        if 'n_init' in model_params:
            clean_param['n_init'] = 20

        return clean_param

    def run_model(self,
                  model_str : str, 
                  params : dict, 
                  scaler_str : str,
                  model_type : str,
                  store_results : bool = False
                  ) -> (str, float):
        
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
    
    def run_redefine(self,
                     super_str : str,
                     super_params : dict,
                     unsup_str : str,
                     unsup_params : dict,
                     scaler_str : str
                     ):
        
        results_df = pd.DataFrame(columns=['Label','SupervisedResult', 'UnsupervisedResult', 'Flagged'], index=self.__IDs)
        results_df['Label'] = self.__Y

        # Run Supervised
        super_info = self.run_model(model_str = super_str,
                                    params = super_params, 
                                    scaler_str = scaler_str,
                                    model_type = "supervised",
                                    store_results = True)
        
        if super_info['error'] is not None:
            return super_info['error'], None, None
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
            return unsup_info['error'], None, None
        else:
            results_df['UnsupervisedResult'] = unsup_info['results']

        # Results
        flagged_ids = self.__eval_misclassed(results_df)

        results_path, metadata_path = self.__get_file_paths()
        

        # TODO: make graph for UI
        # pca / tsne
        plots, plot_random = self.__make_plots(flagged_ids, results_df, scaler_str)

        self.__write_to_files(results_df, super_info, unsup_info, flagged_ids, plot_random, results_path, metadata_path)
        
        return None, flagged_ids, (results_path, metadata_path), plots
    
    def __eval_misclassed(self, results_df):
        flagged_idx = []
        for idx, row in results_df.iterrows():
            if (row['SupervisedResult'] == row['UnsupervisedResult']) and \
            (row['Label'] != row['SupervisedResult']):
                flagged_idx.append(idx)
                results_df.at[idx, 'Flagged'] = True
        return flagged_idx
    
    def __write_to_files(self, results_df, super_info, unsup_info, flagged_ids, plot_random, results_path, metadata_path):
        
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
        return

    def __get_file_paths(self):

        results_path = os.path.join(self.__PATH_OUT, f"results_{self.__file_name}_{self.__timestamp()}.csv")
        metadata_path = os.path.join(self.__PATH_OUT, f"metadata_{self.__file_name}_{self.__timestamp()}.txt")

        return results_path, metadata_path
    
    def __make_plots(self, flagged_ids, results_df, scaler_str):

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

    def __make_plot(self, flagged_ids, results_df, x, plot_type, random_state):
        
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
    
    def __doKFold(self, model, scaler, store_results):
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

    def __doClustering(self,  model, scaler):
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
    def __str_to_num(self, n):
        if n.isnumeric():
            return int(n)
        else:
            try:
                return float(n)
            except:
                return n
    
    def __get_random_seed(self):
        return np.random.randint(0,100000000)
    
    def __accuracy_score(self, ytest, yhat):
        nz = np.flatnonzero(ytest == yhat)
        return len(nz)/len(ytest)

    def __timestamp(self):
        return str(dt.now()).replace(':','-').replace(' ', '_')

    # Getters
    def get_X(self):
        return self.__X.copy()
    
    def get_Y(self):
        return self.__Y.copy()
    
    def get_Y_names(self):
        return self.__Y_names.copy()
    
    def get_IDs(self):
        return self.__IDs.copy()
