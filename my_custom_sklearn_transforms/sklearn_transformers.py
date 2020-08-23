import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import AllKNN
from xgboost import XGBClassifier

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

    
# All sklearn Transforms must have the `transform` and `fit` methods
class Classifier(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y):
        features_knn = [
        'REPROVACOES_DE', 'REPROVACOES_EM', "REPROVACOES_MF", "REPROVACOES_GO",
        "NOTA_DE", "NOTA_EM", "NOTA_MF", "NOTA_GO",
        "INGLES", "H_AULA_PRES", "TAREFAS_ONLINE", "FALTAS", 
        ]

        # Definição da variável-alvo
        target_knn = ["PERFIL"]

        # Preparação dos argumentos para os métodos da biblioteca ``scikit-learn``
        Xlinha = df_data_3[features_knn]
        ylinha = df_data_3[target_knn]

        allknn = AllKNN()
        X_resampled, y_resampled = allknn.fit_resample(Xlinha,ylinha)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=1)        
        model = XGBClassifier()        
        return model
    def transform(self, X):
        return self
