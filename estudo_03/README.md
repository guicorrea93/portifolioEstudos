# Data Science: Regressao com XGBoost (Gradient Boosting)

Estudo de modelos de regressao baseados em Gradient Boosting (XGBoost) para precificacao de automoveis, desenvolvido no curso da [Alura](https://www.alura.com.br/).

---

## Objetivo

Construir modelos de Machine Learning capazes de estimar o valor de venda de automoveis a partir de suas caracteristicas tecnicas e categoricas, aplicando tecnicas de boosting com XGBoost, validacao cruzada e otimizacao de hiperparametros.

## Estudos Realizados

### 1. XGBoost Base (API Scikit-Learn e API Nativa)

Modelo inicial com 100 arvores de boosting — comparacao entre a API wrapper do Scikit-Learn e a API nativa do XGBoost.

- **Dataset:** 10.209 registros (apos remocao de 709 duplicatas), 13 features
- **RMSE Teste:** 3173.81
- **APIs utilizadas:** `XGBRegressor` (Scikit-Learn) e `xgb.train` (nativa)

### 2. Validacao e Early Stopping

Monitoramento de treino vs validacao para identificar overfitting e parada antecipada.

- **Early Stopping:** Melhor iteracao na rodada 37 (de 1000 possiveis)
- **Cross-Validation (5-fold):** 21 rodadas otimas com early stopping
- **RMSE CV Teste:** ~3449

### 3. Otimizacao de Hiperparametros (GridSearchCV)

Busca exaustiva com 18 combinacoes de hiperparametros e 5-fold CV.

- **Melhores Parametros:** colsample_bytree=0.6, max_depth=3, subsample=1
- **Taxas de aprendizagem testadas:** 0.01, 0.1, 0.3, 1.0
- **Melhor learning_rate:** 0.3 (melhor equilibrio convergencia/generalizacao)

### 4. Modelo Final

Modelo otimizado com os melhores hiperparametros, aplicado a dados novos.

- **RMSE Teste Final:** 2912.03
- **Exportacao:** modelo salvo com `joblib` para predicao em producao
- **Predicao:** aplicado a novos automoveis sem valor conhecido

## Tecnicas Utilizadas

| Tecnica | Descricao |
|---------|-----------|
| Limpeza de Dados | Remocao de duplicatas e conversao de tipos (object → category) |
| XGBoost (Scikit-Learn API) | `XGBRegressor` com suporte a variaveis categoricas nativas |
| XGBoost (API Nativa) | `xgb.train` com `DMatrix` para controle granular do treinamento |
| Early Stopping | Parada antecipada baseada em RMSE de validacao |
| Cross-Validation | `xgb.cv` com 5 folds e early stopping |
| GridSearchCV | Busca em grade de colsample_bytree, subsample e max_depth |
| Learning Rate Tuning | Comparacao de taxas 0.01, 0.1, 0.3 e 1.0 com curvas de aprendizado |
| Feature Importance | Ranking de importancia das variaveis pelo XGBoost |
| Serializacao | Exportacao do modelo com `joblib` para uso em producao |

## Pipeline

```
Coleta → Limpeza → Conversao de Tipos → Modelagem XGBoost → Validacao → Otimizacao → Modelo Final → Predicao
```

## Estrutura do Projeto

```
estudo_03/
├── Aula_1.ipynb               # Obtencao e processamento dos dados
├── Aula_2.ipynb               # XGBoost base (APIs Scikit-Learn e nativa)
├── Aula_3.ipynb               # Validacao, early stopping e cross-validation
├── Aula_4.ipynb               # GridSearchCV e tuning de learning rate
├── Aula_5.ipynb               # Modelo final, avaliacao e predicao em dados novos
├── dados_automoveis.csv       # Dataset principal (10.918 registros)
├── novos_automoveis.csv       # Dados novos para predicao
└── README.md
```

## Principais Resultados

**Progressao dos Modelos:**

| Modelo | RMSE Teste |
|--------|------------|
| XGBoost base (100 rounds, lr=0.3) | 3173.81 |
| XGBoost + early stopping (37 rounds) | ~3060 |
| XGBoost + GridSearchCV (depth=3, colsample=0.6) | ~2951 |
| Modelo Final (lr=0.3, otimizado, 170 rounds) | 2912.03 |

**Impacto da Taxa de Aprendizagem:**

| Learning Rate | Rounds Otimos | RMSE Validacao |
|---------------|---------------|----------------|
| 0.01 | 500+ | Convergencia lenta |
| 0.1 | 200+ | ~3119 |
| 0.3 | ~170 | ~2912 |
| 1.0 | ~83 | ~3004 |

## Tecnologias

- **Python 3** | **Pandas** | **NumPy**
- **XGBoost** (XGBRegressor, xgb.train, xgb.cv, DMatrix)
- **Scikit-learn** (train_test_split, GridSearchCV, mean_squared_error)
- **Matplotlib** / **Seaborn** (regplot, plot_importance, curvas de aprendizado)
- **Joblib** (serializacao de modelos)
- **Jupyter Notebook**

## Como Executar

1. Abra os notebooks no [Google Colab](https://colab.research.google.com/) ou Jupyter local
2. Os datasets sao carregados diretamente do GitHub (nao precisa baixar)
3. Execute as celulas sequencialmente (Aula_1 → Aula_5)

## Aprendizados

- XGBoost com suporte nativo a categoricas (`enable_categorical=True`) elimina a necessidade de one-hot encoding manual
- Early stopping previne overfitting automaticamente — o modelo parou na rodada 37 de 1000 possiveis
- `max_depth=3` (arvores rasas) e melhor para boosting do que arvores profundas, pois cada arvore corrige erros residuais
- `colsample_bytree=0.6` introduz aleatoriedade que melhora a generalizacao (similar ao Random Forest)
- Learning rate de 0.3 ofereceu o melhor equilibrio: convergencia rapida sem overfitting precoce
- A API nativa do XGBoost (`xgb.train`) oferece mais controle (evals, callbacks) que o wrapper Scikit-Learn
- Serializar o modelo com `joblib` permite reutiliza-lo sem retreinar
