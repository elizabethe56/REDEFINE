import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

class REDEFINE:
    def __init__(self, data, target_col, id_col):
        # set all data variables
        self.__clean_data(data, target_col, id_col)

        # Constants
        self.__kfolds = 10

        self.__SCALERS = {"Standard" : StandardScaler,
                          "Min-Max" : MinMaxScaler,
                          "Absolute Max" : MaxAbsScaler,
                          "Robust" : RobustScaler}
        self.__MODELS = {"Nearest Neighbor" : KNeighborsClassifier,
                         "Random Forest" : RandomForestClassifier,
                         "KMeans" : KMeans}
        
        return
    
    def __clean_data(self, data, target_col, id_col):
        # Separate data
        self.__IDs = data[id_col]
        Y = data[target_col]
        self.__Y_names = Y.unique()
        self.__Y = Y.values

        data_cols = list(data.columns).copy()
        data_cols.remove(target_col)
        data_cols.remove(id_col)
        self.__X = data[data_cols].values
    
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

    def validate_class_clust(self,
                       model_str : str, 
                       params : dict, 
                       scaler_str : str,
                       model_type : str
                       ) -> (str, float):
        
        info = {'model_str' : model_str, 
                'scaler_str' : scaler_str, 
                'error' : None, 
                'score' : None}

        clean_params = self.__clean_params(model_str, params)
        info['model_params'] = clean_params

        try:
            model = self.__MODELS[model_str](**clean_params)
            if scaler_str == "None":
                scaler = None
            else:
                scaler = self.__SCALERS[scaler_str]()

            if model_type == 'classifier':
                score = self.__doKFold(model, scaler)
            else:
                score = self.__doClustering(model, scaler)
            
            info['score'] = score
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def run_redefine(self,
                     class_str : str,
                     class_params : dict,
                     clust_str : str,
                     clust_params : dict,
                     scaler_str : str
                     ):
        
        return

    
    def __doKFold(self, model, scaler):
        n = len(self.__X)
        idxs = np.linspace(0, n, self.__kfolds+1).astype(int)
        idx = np.arange(0, n)

        self.__kf_random_seed = self.__get_random_seed()
        rs = np.random.RandomState(self.__kf_random_seed)
        rs.shuffle(idx)

        x = self.__X.copy()[idx]
        y = self.__Y.copy()[idx]

        accuracy_scores = np.zeros(self.__kfolds)

        for k, i in enumerate(range(1, len(idxs))):
            idx1 = idxs[i-1]
            idx2 = idxs[i]

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
            
        return np.mean(accuracy_scores)

    def __doClustering(self,  model, scaler):
        x = self.__X.copy()
        y = self.__Y.copy()
        
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
            return self.__accuracy_score(y, yhat)
        else:
            raise Exception("Model did not find the expected number of clusters.")
        
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

    # Getters
    def get_X(self):
        return self.__X.copy()
    
    def get_Y(self):
        return self.__Y.copy()
    
    def get_Y_names(self):
        return self.__Y_names.copy()
    
    def get_IDs(self):
        return self.__IDs.copy()
