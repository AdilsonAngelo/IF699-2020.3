## Naive Bayes Classifier

[dataset](./data/play.csv)

Output:
```
resultado:  Não Jogar

P(False) * P(Chuva|False) * P(Quente|False) * P(Normal|False) * P(Forte|False)
0.5000 * 0.4000 * 0.4000 * 0.4000 * 0.4000 = 0.0128

P(True) * P(Chuva|True) * P(Quente|True) * P(Normal|True) * P(Forte|True)
0.5000 * 0.4000 * 0.2000 * 0.6000 * 0.2000 = 0.0048

```

## How to run
### Python version: 3.8

```sh
$ pip install -r requirements.txt
$ python main.py
```
