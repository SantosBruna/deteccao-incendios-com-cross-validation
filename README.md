# 📊 Projeto: Detecção de Incêndios com Cross Validation

## 📌 Sobre o Projeto

Projeto focado na avaliação de um modelo de **classificação para detecção de incêndios** utilizando a técnica de **Cross Validation (Validação Cruzada)**.

O objetivo é prever a ocorrência de alarmes de incêndio com base em variáveis ambientais coletadas por sensores IoT, utilizando validação cruzada para garantir que o modelo generalize bem para dados não vistos.

---

## 🎯 Objetivos

- Carregar e verificar a base de dados (tipos, dados faltantes e nulos)
- Justificar a escolha do modelo de Machine Learning para o problema
- Separar a base em X e y e instanciar o modelo
- Construir uma pipeline com padronização, balanceamento e treinamento
- Aplicar Cross Validation com KFold para avaliação robusta do modelo
- Avaliar a pontuação de cada fold e a média final

---

## 📁 Estrutura do Projeto

```
├── MOD_35_EXERCICIO.ipynb
├── data/
│   └── smoke_detection_iot.csv
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   └── model_utils.py
└── README.md
```

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Pandas** — Manipulação e análise de dados
- **Scikit-learn** — LogisticRegression, KFold, cross_val_score, train_test_split
- **Imbalanced-learn** — SMOTE, Pipeline com suporte a resamplers
- **Jupyter Notebook** — Ambiente de desenvolvimento

---

## 📊 Descrição dos Dados

O dataset contém leituras de sensores ambientais IoT para detecção de incêndios, com as seguintes variáveis:

| Variável | Descrição |
|---|---|
| UTC | Tempo em segundos UTC |
| Temperature[C] | Temperatura do ar em graus Celsius |
| Humidity[%] | Umidade do ar em porcentagem |
| TVOC[ppb] | Total de compostos orgânicos voláteis (ppb) |
| eCO2[ppm] | Concentração equivalente de CO2 (ppm) |
| Raw H2 | Hidrogênio molecular bruto, não compensado |
| Raw Ethanol | Etanol gasoso bruto |
| Pressure[hPA] | Pressão do ar em hectopascais |
| PM1.0 | Material particulado < 1,0 µm |
| PM2.5 | Material particulado entre 1,0 µm e 2,5 µm |
| NC0.5 | Concentração numérica de partículas < 0,5 µm |
| NC1.0 | Concentração numérica de partículas entre 0,5 µm e 1,0 µm |
| NC2.5 | Concentração numérica de partículas entre 1,0 µm e 2,5 µm |
| CNT | Contador de amostras |
| **Fire_Alarm** | **Variável alvo — 1 se houver incêndio, 0 caso contrário** |

---

## 📈 Etapas Realizadas

### Etapa 1 — Carregamento e Verificação dos Dados

- Leitura da base com `pd.read_csv`
- Verificação de tipos com `.dtypes`
- Verificação de nulos com `.isnull().sum()` e `.isna().sum()`
- Renomeação da coluna `Fire Alarm` para `Fire_Alarm` para facilitar o acesso

---

### Etapa 2 — Escolha do Modelo

O target `Fire_Alarm` é uma variável binária (0 ou 1), o que caracteriza um problema de **classificação**. Os modelos candidatos foram:

- Regressão Logística
- Árvore de Decisão
- Random Forest

**Modelo escolhido:** Regressão Logística como ponto de partida — caso a avaliação não apresente bom desempenho, o Random Forest seria a próxima escolha.

---

### Etapa 3 — Separação e Pipeline

- Separação de X e y com `data_utils.separar_x_y`
- Criação da pipeline com `model_utils.criar_pipeline_regressao_logistica`, contendo:
  - `StandardScaler` — padronização das features
  - `SMOTE` — balanceamento das classes
  - `LogisticRegression` — treinamento do modelo

---

### Etapa 4 — Cross Validation

- Aplicação do `KFold` com 5 folds, shuffle ativado
- Avaliação com `cross_val_score` usando a métrica `accuracy`
- O SMOTE é aplicado apenas nos dados de treino de cada fold, sem contaminação do conjunto de teste

---

### Etapa 5 — Resultados

| Métrica | Valor |
|---|---|
| Pontuação máxima (fold) | ~0.9827 |
| Média dos folds | próxima de 1.0 |
| Desvio padrão | muito baixo |

**Conclusão:** todos os 5 folds apresentaram desempenho excelente, com pontuações muito próximas entre si, indicando um modelo estável e bem generalizado.

---

## 🔍 Principais Insights

1. O modelo de Regressão Logística apresentou **alta acurácia** já na primeira tentativa, não sendo necessário migrar para modelos mais complexos
2. O **desvio padrão muito baixo** entre os folds indica que o modelo é **consistente** — não depende de uma divisão específica dos dados para performar bem
3. A pipeline garantiu que o **SMOTE nunca vazou para os dados de teste**, evitando data leakage e tornando a avaliação confiável

---

## 👩‍💻 Autora

**Bruna S. R. Santos**

- 🔗 [LinkedIn](https://www.linkedin.com/in/brunasrsantos)
- 📧 brunasrsantos@gmail.com

---

## 📝 Licença

Este projeto está licenciado sob a **MIT License**.
