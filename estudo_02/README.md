# Data Science: Regressao com Arvores de Decisao e Random Forest

Estudo de modelos de regressao baseados em arvores para precificacao do custo de entrega de esculturas, desenvolvido no curso da [Alura](https://www.alura.com.br/).

**[Ver pagina do estudo](https://guicorrea93.github.io/PortifolioProjetos/projeto_alura_02/)**

---

## Objetivo

Construir modelos de Machine Learning capazes de estimar o custo de entrega de esculturas artisticas com base em caracteristicas fisicas da peca, tipo de transporte, reputacao do artista e outras variaveis logisticas, aplicando tecnicas de arvores de regressao e ensemble.

## Estudos Realizados

### 1. Arvore de Decisao (Decision Tree)

Modelo inicial com arvore de decisao — identificacao de overfitting e otimizacao via GridSearchCV.

- **Dataset:** 4.462 registros, 30 features (apos feature engineering)
- **Modelo Base:** R² Treino: 1.000 | R² Teste: 0.661 (overfitting severo)
- **Modelo Otimizado:** R² Treino: 0.871 | R² Teste: 0.754
- **Melhores Parametros:** max_depth=10, min_samples_leaf=10, min_samples_split=2

### 2. Random Forest (Floresta Aleatoria)

Ensemble de 100 arvores com validacao OOB e otimizacao por GridSearchCV.

- **Modelo Base:** R² Treino: 0.977 | R² Teste: 0.825 | OOB R²: 0.828
- **Modelo Otimizado:** R² Treino: 0.966 | R² Teste: 0.826
- **Melhores Parametros:** max_depth=None, max_leaf_nodes=550, min_samples_leaf=2
- **Cross-Validation (3-fold):** RMSE Teste: 1143.54

## Tecnicas Utilizadas

| Tecnica | Descricao |
|---------|-----------|
| Feature Engineering | Extracao de dia, mes e ano de datas; calculo de diferenca entre datas |
| One-Hot Encoding | Conversao de 8 variaveis categoricas em dummies com drop_first=True |
| Decision Tree | Arvore de regressao com criterio MSE para splits |
| Random Forest | Ensemble bagging com 100 arvores e avaliacao OOB |
| GridSearchCV | Busca exaustiva de hiperparametros com cross-validation |
| Cross-Validation | KFold com 3 splits para avaliar generalizacao |
| Feature Importance | Ranking de importancia das variaveis pelo modelo |

## Pipeline

```
Coleta → Pre-processamento → Feature Engineering → Modelagem → Otimizacao → Predicao
```

## Estrutura do Projeto

```
estudo_02/
├── Projeto_Regressao.ipynb                  # Notebook de trabalho
├── Final - Projeto_Regressao.ipynb          # Notebook final completo
├── entregas.csv                              # Dataset de treino (4.462 registros)
├── teste_entregas.csv                        # Dataset de teste (1.434 registros)
├── index.html                                # Pagina interativa do estudo
└── README.md
```

## Principais Resultados

**Progressao dos Modelos:**

| Modelo | R² Treino | R² Teste | RMSE Teste |
|--------|-----------|----------|------------|
| Arvore (base) | 1.000 | 0.661 | 1544.65 |
| Arvore (otimizada) | 0.871 | 0.754 | 1315.20 |
| Random Forest (base) | 0.977 | 0.825 | 1108.02 |
| Random Forest (otimizada) | 0.966 | 0.826 | 1106.16 |

**Top 5 Features Mais Importantes:**

| Feature | Importancia |
|---------|-------------|
| preco_escultura | 38.49% |
| reputacao_artista | 29.56% |
| preco_base_envio | 14.94% |
| altura | 4.29% |
| largura | 3.78% |

## Tecnologias

- **Python 3** | **Pandas** | **NumPy**
- **Scikit-learn** (DecisionTreeRegressor, RandomForestRegressor, GridSearchCV, cross_validate, KFold)
- **Matplotlib** / **Seaborn** (barh, feature importances)
- **Jupyter Notebook**

## Como Executar

1. Abra os notebooks no [Google Colab](https://colab.research.google.com/) ou Jupyter local
2. Os datasets sao carregados diretamente do GitHub (nao precisa baixar)
3. Execute as celulas sequencialmente

## Aprendizados

- Arvores de decisao sem restricoes sofrem overfitting severo (R² treino = 1.0, teste = 0.66)
- Hiperparametros como max_depth e min_samples_leaf sao essenciais para regularizacao
- Random Forest reduz variancia significativamente: RMSE caiu de 1544 para 1108
- GridSearchCV com cross-validation encontra combinacoes otimas de hiperparametros sistematicamente
- Feature importance revela que preco da escultura e reputacao do artista dominam a predicao (~68% juntos)
- OOB score do Random Forest e uma estimativa confiavel da performance sem precisar de conjunto de teste separado
