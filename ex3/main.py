import os
import pandas as pd
from bayes import NaiveBayesClassifier


def main():
    curr_dir = os.path.dirname(__file__)
    csv_file = os.path.join(curr_dir, 'data/play.csv')

    test_case = {
        'Tempo': 'Chuva',
        'Temperatura': 'Quente',
        'Humidade': 'Normal',
        'Vento': 'Forte'
    }

    df = pd.read_csv(csv_file, index_col='Dia')
    X, y = df.loc[:, df.columns != 'Jogar'], df['Jogar']

    clf = NaiveBayesClassifier()
    clf.fit(X, y)
    print('resultado: ', 'Jogar' if clf.predict(test_case) else 'Não Jogar')
    print()
    print(clf.get_probs_str(test_case))


if __name__ == "__main__":
    main()
