import os
import pandas as pd
from decision_tree import DecisionTreeClassifier


def main():
    curr_dir = os.path.dirname(__file__)
    csv_file = os.path.join(curr_dir, 'data/play.csv')

    df = pd.read_csv(csv_file, index_col='Dia')
    X, y = df.loc[:, df.columns != 'Jogar'], df['Jogar']

    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    print(clf.rules())


if __name__ == "__main__":
    main()
