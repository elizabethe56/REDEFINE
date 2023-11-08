import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# TODO:
# Replicability? How to use random states without making it too clunky?
# Is it better to keep classifier/cluster alg separate, or combine them?
    # Will lead to class/clust flags and more if statements for diverging paths
# What to do when the clustering algorithm fails hard enough to group two clusters and split the other?
# Should we be doing PCA?

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
        self.__MODELS = {"Nearest Neighbor" : NearestNeighbors,
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
    
    def __clean_params(self, params):
        return { key:self.__str_to_num(val) 
                 for (key, val) in params.items() 
                 if (val != "") and (val is not None) }

    def validate_class_clust(self, 
                            model_str : str, 
                            params : dict, 
                            scaler_str : str,
                            model_type : str
                            ) -> (str, float):

        params = self.__clean_params(params)

        try:
            # "Nearest Neighbor", "Random Forest"
            model = self.__MODELS[model_str](**params)
            model.fit(self.__X[0:len(self.__Y_names) + 1], self.__Y[0:len(self.__Y_names) + 1])
        except Exception as e:
            return str(e), None
        
        scaler = self.__SCALERS[scaler_str]()

        try:
            results = self.__doKFold(model, scaler, model_type)
            return None, results
        except KeyError as e:
            return "KeyError: Check the number of clusters", None
    
    def __doKFold(self, model, scaler, model_type):
        n = len(self.__X)
        idxs = np.linspace(0, n, self.__kfolds+1).astype(int)
        idx = np.arange(0, n)
        np.random.shuffle(idx)

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

            scale = scaler
            xtrain_scale = scale.fit_transform(xtrain)
            xtest_scale = scale.transform(xtest)

            m = model

            m.fit(xtrain_scale, ytrain)
            yhat = m.predict(xtest_scale)

            # relabeling for clustering
            if model_type == 'cluster':
                yhat_train = m.predict(xtrain_scale)
                label_map = {}
                for i in range(len(self.__Y_names)):
                    where_i = np.where(ytrain == self.__Y_names[i])
                    val_i = np.bincount(yhat_train[where_i]).argmax()
                    label_map[val_i] = self.__Y_names[i]

                if len(label_map) == len(self.__Y_names):
                    yhat = np.vectorize(label_map.__getitem__)(yhat)
            
            accuracy_scores[k] = self.__accuracy_score(ytest, yhat)
            
        return np.mean(accuracy_scores)

    def __accuracy_score(self, ytest, yhat):
        nz = np.flatnonzero(ytest == yhat)
        return len(nz)/len(ytest)
        
    # Misc
    def __str_to_num(self, n):
        if n.isnumeric():
            return int(n)
        else:
            try:
                return float(n)
            except:
                return n

    # Getters
    def get_X(self):
        return self.__X.copy()
    
    def get_Y(self):
        return self.__Y.copy()
    
    def get_Y_names(self):
        return self.__Y_names.copy()
    
    def get_IDs(self):
        return self.__IDs.copy()
