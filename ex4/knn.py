import pandas as pd


class KNNClassifier():
    def __init__(self, k: int):
        self.k = k

    def hamming_distance(self, x1: pd.Series, x2: pd.Series):
        res = x1.eq(x2).apply(lambda x: 0 if x else 1)
        return sum(res)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def predict(self, x: dict):
        x = pd.Series(x)
        distances = []
        for index, row in self.X.iterrows():
            distances.append({'dia': index,
                              'dist': self.hamming_distance(x, row),
                              'class': self.y.loc[index]})

        distances.sort(key=lambda x: x['dist'])

        self.knn = pd.DataFrame(distances)
        self.knn.index = self.knn['dia']
        self.knn.drop(columns=['dia'], inplace=True)

        counts = dict()
        for d in distances[:self.k]:
            counts[d['class']] = counts.get(d['class'], 0) + 1

        return max(counts, key=counts.get)
