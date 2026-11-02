# Sistemas Inteligentes — Lista de Projetos

Repositório desenvolvido como parte da **1ª Lista de Projetos** da disciplina **PAM0466 — Sistemas Inteligentes**, da Universidade Federal Rural do Semi-Árido (UFERSA), campus Pau dos Ferros, referente ao semestre 2026.2.

O trabalho contempla a implementação e a avaliação de quatro algoritmos de aprendizado de máquina, envolvendo problemas de regressão e classificação supervisionada.

## Informações Acadêmicas

- **Instituição:** Universidade Federal Rural do Semi-Árido (UFERSA)
- **Disciplina:** PAM0466 — Sistemas Inteligentes
- **Professor:** Pedro Thiago Valério de Sousa
- **Aluno:** Pedro Lucas de Sousa Ferreira
- **Semestre:** 2026.2

## Tecnologias Utilizadas

- **Python:** linguagem principal de desenvolvimento.
- **PyTorch:** implementação dos modelos de regressão linear, KNN e árvore de decisão.
- **Scikit-learn:** pré-processamento, validação cruzada, métricas e regressão logística.
- **Pandas e NumPy:** manipulação e análise dos dados.
- **Matplotlib e Seaborn:** visualização dos resultados.
- **Graphviz:** visualização gráfica da árvore de decisão.
- **Jupyter Notebook:** desenvolvimento e execução dos experimentos.

## Projetos Desenvolvidos

### 1. Regressão Linear Múltipla — Previsão de Produção de Energia

Implementação de um modelo de regressão linear múltipla utilizando PyTorch para prever a produção elétrica de uma usina de ciclo combinado.

Foi utilizado o dataset *Combined Cycle Power Plant*, contendo 9.568 amostras e quatro variáveis independentes: temperatura ambiente, vácuo de escape, pressão atmosférica e umidade relativa.

O projeto contempla análise de correlação, investigação de multicolinearidade, padronização, treinamento e avaliação do modelo.

**Resultados no conjunto de teste:**

| Métrica | Resultado |
|---|---:|
| R² | 0,9301 |
| RMSE | 4,5016 MW |
| MAE | 3,5954 MW |

**Arquivos:**

- [Notebook — Regressão Linear](notebooks/01_regressao_linear.ipynb)
- [Relatório — Regressão Linear](relatorios/01_regressao_linear.md)

### 2. Regressão Logística — Previsão de Doenças Cardíacas

Desenvolvimento de um classificador binário utilizando regressão logística para identificar a presença ou ausência de doenças cardíacas com base em características clínicas.

Foi utilizado o dataset *Heart Disease Cleveland*, contendo 303 registros de pacientes.

A implementação utiliza `LogisticRegression` do scikit-learn, conforme os parâmetros definidos no enunciado da atividade.

O projeto contempla tratamento de valores ausentes, codificação de atributos categóricos, padronização, treinamento e avaliação por métricas de classificação.

**Resultados no conjunto de teste:**

| Métrica | Resultado |
|---|---:|
| Acurácia | 86,89% |
| Precisão | 81,25% |
| Recall | 92,86% |
| F1-score | 0,8667 |
| ROC-AUC | 0,9578 |

O modelo convergiu em seis iterações, utilizando o solver `liblinear`.

**Arquivos:**

- [Notebook — Regressão Logística](notebooks/02_regressao_logistica.ipynb)
- [Relatório — Regressão Logística](relatorios/02_regressao_logistica.md)

### 3. K-Nearest Neighbors (KNN) — Classificação da Qualidade de Vinhos

Implementação do algoritmo KNN utilizando operações tensoriais do PyTorch para classificar vinhos tintos em três categorias de qualidade: Ruim, Médio e Bom.

Foi utilizado o dataset *Wine Quality — Red Wine*, contendo originalmente 1.599 amostras e 11 atributos físico-químicos.

Foram realizadas análise exploratória, remoção de registros duplicados, padronização por Z-score e validação cruzada estratificada de cinco folds para selecionar o número de vizinhos.

**Resultados:**

| Métrica | Resultado |
|---|---:|
| Melhor k | 11 |
| Acurácia média na validação | 72,68% |
| Acurácia no teste | 66,54% |
| F1-score macro no teste | 0,4457 |

O modelo apresentou maior dificuldade na diferenciação entre vinhos das classes Ruim e Médio e não conseguiu identificar corretamente as amostras da classe Bom no conjunto de teste.

**Arquivos:**

- [Notebook — KNN](notebooks/03_knn.ipynb)
- [Relatório — KNN](relatorios/03_knn.md)

### 4. Árvore de Decisão — Classificação de Risco de Crédito

Implementação de uma árvore de decisão utilizando PyTorch para classificar solicitantes de crédito nas categorias Bom Risco e Mau Risco.

Foi utilizado o dataset *Statlog (German Credit Data)*, contendo 1.000 registros e 20 atributos cadastrais e financeiros.

O algoritmo foi desenvolvido com construção recursiva de nós, critérios de divisão por Gini e Entropia, controle de profundidade e identificação da importância dos atributos.

A seleção dos hiperparâmetros foi realizada utilizando validação cruzada estratificada de cinco folds.

**Resultados:**

| Métrica | Resultado |
|---|---:|
| Critério selecionado | Entropia |
| Profundidade máxima | 7 |
| Acurácia média na validação | 72,12% |
| Acurácia no teste | 64,50% |
| Recall — Mau Risco | 50,00% |
| F1-score — Mau Risco | 0,4580 |

A árvore apresentou limitações na identificação de clientes de mau risco, apesar de permitir a interpretação das decisões por meio das regras e da importância dos atributos.

**Arquivos:**

- [Notebook — Árvore de Decisão](notebooks/04_arvore_decisao.ipynb)
- [Relatório — Árvore de Decisão](relatorios/04_arvore_decisao.md)

## Estrutura do Repositório

```text
Sistemas-Inteligentes/
│
├── notebooks/
│   ├── 01_regressao_linear.ipynb
│   ├── 02_regressao_logistica.ipynb
│   ├── 03_knn.ipynb
│   └── 04_arvore_decisao.ipynb
│
├── relatorios/
│   ├── 01_regressao_linear.md
│   ├── 02_regressao_logistica.md
│   ├── 03_knn.md
│   └── 04_arvore_decisao.md
│
├── figuras/
│   └── Gráficos e visualizações dos experimentos
│
├── .gitignore
├── requirements.txt
└── README.md
```

Os notebooks contêm os códigos e experimentos realizados, enquanto os relatórios apresentam a metodologia, a interpretação dos resultados e as conclusões de cada problema.

## Como Executar

**1. Clone o repositório:**

```bash
git clone URL_DO_REPOSITORIO
cd Sistemas-Inteligentes
```

Substitua `URL_DO_REPOSITORIO` pelo endereço do repositório no GitHub.

**2. Crie um ambiente virtual:**

```bash
python -m venv .venv
```

**3. Ative o ambiente virtual no Windows:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**4. Instale as dependências:**

```bash
python -m pip install -r requirements.txt
```

Para a visualização da árvore de decisão, também é necessário ter o Graphviz instalado no sistema e o executável `dot` disponível no PATH.

**5. Execute os notebooks:**

Abra o projeto no Visual Studio Code, selecione o interpretador Python do ambiente virtual e execute os arquivos `.ipynb` utilizando a extensão Jupyter.

Os datasets são carregados a partir dos respectivos repositórios públicos, sendo necessária conexão com a internet durante o carregamento inicial.

## Organização dos Experimentos

Os projetos foram desenvolvidos com foco na compreensão dos algoritmos, na reprodutibilidade dos experimentos e na interpretação dos resultados.

As implementações incluem técnicas de pré-processamento, separação dos conjuntos de treinamento e teste, validação cruzada quando solicitada e avaliação por métricas adequadas a cada problema.

Os resultados, inclusive as limitações identificadas, foram documentados individualmente nos relatórios Markdown.

## Referências dos Datasets

- [Combined Cycle Power Plant — UCI](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)
- [Heart Disease — UCI](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [Wine Quality — UCI](https://archive.ics.uci.edu/dataset/186/wine+quality)
- [Statlog German Credit Data — UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
