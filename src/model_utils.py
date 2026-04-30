from sklearn.cluster import KMeans
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

def criar_pipeline_kmeans(n_clusters=5, random_state=42):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state))
    ])
    
    return pipeline

def treinar_pipeline(df, colunas, pipeline):
    X = df[colunas]
    pipeline.fit(X)
    return pipeline

def criar_pipeline_regressao_logistica(
    random_state=42,
    smote_k_neighbors=5,
    C=1.0,
    max_iter=1000,
    class_weight=None
):
    """
    Cria um pipeline com StandardScaler, SMOTE e Regressão Logística.

    Parâmetros:
    -----------
    random_state : int
        Semente para reprodutibilidade (SMOTE e modelo).
    smote_k_neighbors : int
        Número de vizinhos usados pelo SMOTE.
    C : float
        Inverso da força de regularização do modelo.
    max_iter : int
        Número máximo de iterações do solver.
    class_weight : dict ou 'balanced' ou None
        Peso das classes no modelo. Útil se não quiser usar SMOTE.

    Retorna:
    --------
    pipeline : imblearn.pipeline.Pipeline
    """
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('balanceamento', SMOTE(
            k_neighbors=smote_k_neighbors,
            random_state=random_state
        )),
        ('modelo', LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state
        ))
    ])

    return pipeline

from sklearn.model_selection import KFold, cross_val_score
import numpy as np

def criar_cross_validation(
    pipeline,
    X,
    y,
    folds=5,
    random_state=5,
    scoring='accuracy'
):
    """
    Executa cross-validation com KFold em um pipeline.

    Parâmetros:
    -----------
    pipeline : Pipeline
        Pipeline já criado (com scaler, balanceamento e modelo).
    X : array-like
        Features.
    y : array-like
        Target.
    folds : int
        Número de folds do KFold. Padrão: 5.
    random_state : int
        Semente para reprodutibilidade do KFold. Padrão: 5.
    scoring : str
        Métrica de avaliação. Padrão: 'accuracy'.

    Retorna:
    --------
    dict com KFold, pontuações por fold, média e desvio padrão.
    """
    crossvalidation = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state
    )

    pontuacoes = cross_val_score(
        pipeline,
        X,
        y,
        cv=crossvalidation,
        scoring=scoring
    )

    print(f"Pontuações por fold : {pontuacoes}")
    print(f"Média               : {pontuacoes.mean():.4f}")
    print(f"Desvio padrão       : {pontuacoes.std():.4f}")

    return {
        'kfold': crossvalidation,
        'pontuacoes': pontuacoes,
        'media': pontuacoes.mean(),
        'desvio_padrao': pontuacoes.std()
    }

def extrair_resultados_kmeans_pipeline(pipeline, X):
    """
    Extrai os labels previstos e os centroides (na escala original)
    de um pipeline contendo um modelo KMeans.

    Parâmetros:
    - pipeline: objeto sklearn Pipeline já treinado.
        Deve conter:
            - 'scaler': etapa de padronização (ex: StandardScaler)
            - 'kmeans': modelo KMeans treinado

    - X: DataFrame ou array com os dados de entrada.
        Deve conter exatamente as mesmas colunas utilizadas no treinamento.
        Cada coluna representa uma feature do modelo, por exemplo:
            - 'Age': idade do cliente
            - 'Annual Income (k$)': renda anual
            - 'Spending Score': score de consumo

    Retorno:
    - labels: array com o cluster atribuído a cada linha de X
        Exemplo: [0, 1, 1, 2, 0, ...]

    - centroides: array com as coordenadas dos centroides
        no espaço original (antes da padronização).
        Cada linha representa um cluster e cada coluna corresponde
        a uma feature de X.

        Exemplo (3 clusters, 3 variáveis):
        [
            [idade, renda, score],
            [idade, renda, score],
            [idade, renda, score]
        ]
    """

    kmeans = pipeline.named_steps["kmeans"]
    scaler = pipeline.named_steps["scaler"]

    # Gera os labels com base nos dados de entrada
    labels = pipeline.predict(X)

    # Converte os centroides da escala padronizada para a escala original
    centroides = scaler.inverse_transform(kmeans.cluster_centers_)

    return labels, centroides