from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


class REDEFINE:
    def __init__(self, data, target_col, id_col):
        self.__clean_data(data, target_col, id_col)
        self.__models = {"Nearest Neighbor" : NearestNeighbors,
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

    def validate_classifier(self, 
                            model_str : str, 
                            params : dict, 
                            scaler_str : str
                            ) -> (str, float):

        params = self.__clean_params(params)

        try:
            # "Nearest Neighbor", "Random Forest"
            model = self.__models[model_str](**params)
            model.fit(self.__X[0:1], self.__Y[0:1])
        except TypeError as e:
            return str(e), None
        except Exception as e:
            print(e)
            return str(e), None

        return None, 7690
    
    def validate_cluster_alg(self, model, params, scaler):
        # TODO
        print(model)
        for key, val in params.items():
            print(key, val)
        return None, None

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
