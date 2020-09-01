import pandas as pd
from anytree import Node, RenderTree


class DecisionTreeClassifier:
    def _col_gini(self, x: pd.Series, y: pd.Series):
        '''calcula grau de impureza da coluna x'''

        res = dict()
        for v in pd.unique(x):
            pos = len(y[y & (x == v)])
            neg = len(y[~y & (x == v)])

            res[v] = 1 - (pos/(pos+neg)) ** 2 - (neg/(pos+neg)) ** 2

        return sum(v * len(x[x == k]) / len(x) for k, v in res.items())

    def _gini(self, X: pd.DataFrame, y: pd.Series):
        '''gera tuplas do formato: (column, gini_index)'''
        for col in X:
            yield (col, self._col_gini(X[col], y))

    def _gen_tree(self, X: pd.DataFrame, y: pd.Series, parent_col='root', parent_val=''):   # noqa
        '''retorna o no raiz da arvore de decisao'''
        # se todos forem da mesma classe retorna nó folha
        if y.all() or (~y).all():
            return Node(parent_col, val=parent_val, target=y.all())                         # noqa

        # pega coluna com menor entropia
        this_col, _ = sorted([g for g in self._gini(X, y)], key=lambda tup: tup[1])[0]      # noqa

        children = [
            self._gen_tree(X[X[this_col] == val],
                           y[X[this_col] == val], this_col, val)

            for val in pd.unique(X[this_col])
        ]

        return Node(parent_col, children=children, val=parent_val, target='')

    def fit(self, dataset: pd.DataFrame, target: pd.Series):
        '''constroi a arvore de decisao'''
        self.tree = RenderTree(self._gen_tree(dataset, target))

    def rules(self):
        '''retorna uma string com a representacao da arvore de decisao'''
        if self.tree is not None:

            # print(self.tree)
            return '\n'.join(f'{pre}{node.name}({node.val}): {node.target}'
                             for pre, _, node in self.tree)
