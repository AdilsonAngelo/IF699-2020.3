import pandas as pd
from anytree import Node, RenderTree


class DecisionTreeClassifier:
    def _col_gini(self, x: pd.Series, y: pd.Series):
        res = dict()
        for v in pd.unique(x):
            pos = len(y[y & (x == v)])
            neg = len(y[~y & (x == v)])

            res[v] = 1 - (pos/(pos+neg)) ** 2 - (neg/(pos+neg)) ** 2

        return sum(v * len(x[x == k]) / len(x) for k, v in res.items())

    def _gini(self, X: pd.DataFrame, y: pd.Series):
        for col in X:
            yield (col, self._col_gini(X[col], y))

    def _fit(self, X: pd.DataFrame, y: pd.Series, parent_col='root', parent_val=''):    # noqa
        if y.all() or (~y).all():
            return Node(parent_col, val=parent_val, target=y.all())                     # noqa

        this_col, _ = sorted([g for g in self._gini(X, y)], key=lambda tup: tup[1])[0]       # noqa

        children = []
        for val in pd.unique(X[this_col]):
            child = self._fit(X[X[this_col] == val],
                              y[X[this_col] == val], this_col, val)
            children.append(child)

        return Node(parent_col, children=children, val=parent_val, target='')

    def fit(self, dataset: pd.DataFrame, target: pd.Series):
        self.tree = RenderTree(self._fit(dataset, target))

    def rules(self):
        if self.tree is not None:

            # print(self.tree)
            for pre, fill, node in self.tree:
                print(f'{pre}{node.name}({node.val}): {node.target}')
