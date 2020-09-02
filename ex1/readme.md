## Decision Tree Classifier

Simple Decision Tree Classifier using Gini. Works only with the given [dataset](./data/play.csv)

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