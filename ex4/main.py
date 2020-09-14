import os
import pandas as pd
from knn import KNNClassifier


def main():
    curr_dir = os.path.dirname(__file__)
    csv_file = os.path.join(curr_dir, 'data/play.csv')

    test = pd.Series({
        'Tempo': 'Chuva',
        'Temperatura': 'Quente',
        'Humidade': 'Normal',
        'Vento': 'Forte'
    })

    df = pd.read_csv(csv_file, index_col='Dia')
    X, y = df.loc[:, df.columns != 'Jogar'], df['Jogar']

    clf = KNNClassifier(k=1)
    clf.fit(X, y)

    print(f'RESULT k = {clf.k} ::',
          'Jogar' if clf.predict(test) else 'Não Jogar')
    clf.k = 3
    print(f'RESULT k = {clf.k} ::',
          'Jogar' if clf.predict(test) else 'Não Jogar')
    print()
    print('DISTANCES')
    print(clf._gen_distances(test))


if __name__ == "__main__":
    main()
