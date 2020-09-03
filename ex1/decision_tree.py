import pandas as pd
from math import log2
from anytree import Node, RenderTree


class DecisionTreeClassifier:

    def _entropy(self, pos_prob, neg_prob):
        pos_w = pos_prob * log2(pos_prob) if pos_prob > 0 else 0
        neg_w = neg_prob * log2(neg_prob) if neg_prob > 0 else 0
        return -(pos_w + neg_w)

    def _col_info_gain(self, x: pd.Series, y: pd.Series, ds_entropy):
        '''calcula ganho de informacao da coluna x'''

        res = []
        for v in pd.unique(x):
            pos = len(y[y & (x == v)])
            neg = len(y[~y & (x == v)])
            total = len(y[x == v])

            res.append(self._entropy(pos/total, neg/total) * total/len(y))

        return ds_entropy - sum(res)

    def _all_info_gain(self, X: pd.DataFrame, y: pd.Series, ds_entropy):
        '''gera tuplas do formato: (column, info_gain)'''
        for col in X:
            yield (col, self._col_info_gain(X[col], y, ds_entropy))

    def _gen_tree(self, X: pd.DataFrame, y: pd.Series, parent_col='root', parent_val=''):   # noqa
        '''funcao recursiva: retorna o no da arvore de decisao e suas ramificacoes'''       # noqa

        # se todos forem da mesma classe retorna nó folha
        if y.all() or (~y).all():
            return Node(parent_col, val=parent_val, target=y.all())                         # noqa

        curr_entropy = self._entropy(len(y[y])/len(y), len(y[~y])/len(y))

        chosen_col, _ = sorted(self._all_info_gain(X, y, curr_entropy),
                               key=lambda x: x[1], reverse=True)[0]

        children = [
            self._gen_tree(X[X[chosen_col] == val],
                           y[X[chosen_col] == val], chosen_col, val)

            for val in pd.unique(X[chosen_col])
        ]

        return Node(parent_col, children=children, val=parent_val, target='')

    def fit(self, X: pd.DataFrame, y: pd.Series):
        '''constroi a arvore de decisao'''
        self.tree = RenderTree(self._gen_tree(X, y))

    def rules(self):
        '''retorna uma string com a representacao da arvore de decisao'''
        if self.tree is not None:
            return '\n'.join(f'{pre}{node.name}({node.val}): {node.target}'
                             for pre, _, node in self.tree)
