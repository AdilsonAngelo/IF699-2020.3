from math import prod
from pandas import DataFrame, Series


class NaiveBayesClassifier:
    def fit(self, X: DataFrame, y: Series):
        self.X = X
        self.y = y
        self.class_probs = self._class_probabilities(X, y)
        self.feat_probs = self._feature_probabilities(X, y)

    def predict(self, x: dict):
        probs = {
            _class: self.class_probs[_class] * prod(
                self.feat_probs[col][val][_class]

                for col, val in x.items()
            )
            for _class in self.y.unique()
        }
        return max(probs, key=probs.get)

    def _class_probabilities(self, X: DataFrame, y: Series):
        '''retorna um dicionario com as probabilidades de cada classe'''
        return {
            val: len(y[y == val]) / len(y)

            for val in y.unique()
        }

    def _feature_probabilities(self, X: DataFrame, y: Series):
        '''
        retorna um dicionario aninhado com as probabilidades de cada valor
        de cada coluna para cada classe
        '''
        return {
            col: {
                col_val: {
                    _class: (1+len(y[(y == _class) & (X[col] == col_val)])) /
                    len(y[y == _class])

                    for _class in y.unique()

                } for col_val in X[col].unique()

            } for col in X
        }
