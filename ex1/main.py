import pandas as pd
from decision_tree import DecisionTreeClassifier


def main():
    df = pd.read_csv('./data/play.csv', index_col='Dia')
    X, y = df.loc[:, df.columns != 'Jogar'], df['Jogar']
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    clf.rules()


if __name__ == "__main__":
    main()
