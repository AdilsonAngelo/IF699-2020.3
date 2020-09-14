import pandas as pd


class KNNClassifier():
    def __init__(self, k: int):
        self.k = k

    def hamming_distance(self, x1: pd.Series, x2: pd.Series):
        return x1.eq(x2).apply(lambda x: 0 if x else 1).sum()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def predict(self, x: pd.Series):
        distances = self._gen_distances(x)
        return distances.iloc[:self.k]['class'].value_counts().idxmax()

    def _gen_distances(self, x: pd.Series):
        distances = []
        for index, row in self.X.iterrows():
            distances.append({self.X.index.name: index,
                              'dist': self.hamming_distance(x, row),
                              'class': self.y.loc[index]})

        distances.sort(key=lambda x: x['dist'])

        distances = pd.DataFrame(distances)
        distances.index = distances[self.X.index.name]
        distances.drop(columns=[self.X.index.name], inplace=True)
        return distances
