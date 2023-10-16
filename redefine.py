
from sklearn.preprocessing import StandardScaler

class REDEFINE:
    def __init__(self, data, target_col, id_col):
        self.__clean_data(data, target_col, id_col)
        return
    
    def __clean_data(self, data, target_col, id_col):
        print(type(data))
        print(type(target_col))
        print(data[target_col])
        # Separate data
        self.__IDs = data[id_col]
        Y = data[target_col]
        self.__Y_names = Y.unique()
        self.__Y = Y.values

        data_cols = list(data.columns).copy()
        data_cols.remove(target_col)
        data_cols.remove(id_col)
        X = data[data_cols].values

        # Standardize X
        scaler = StandardScaler()
        self.__X = scaler.fit_transform(X)

    # Getters
    def get_X(self):
        return self.__X.copy()
    
    def get_Y(self):
        return self.__Y.copy()
    
    def get_Y_names(self):
        return self.__Y_names.copy()
    
    def get_IDs(self):
        return self.__IDs.copy()
