## Decision Tree Classifier

Simple Decision Tree Classifier using Information Gain. Works only with the given [dataset](./data/play.csv)

Output:
```
root():
├── Tempo(Sol): 
│   ├── Temperatura(Quente): False
│   ├── Temperatura(Normal): False
│   └── Temperatura(Frio): True
├── Tempo(Coberto): True
└── Tempo(Chuva): 
    ├── Temperatura(Normal): True
    └── Temperatura(Frio): False
```

## How to run
### Python version: 3.8

Run
```sh
$ make build
$ make run
```
Or
```sh
$ pip install -r requirements
$ python main.py
```
