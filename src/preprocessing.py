import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class CollinearFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformateur Scikit-Learn personnalisé pour regrouper les variables 
    fortement corrélées (|r| >= threshold) et ne conserver que la variable 
    la plus corrélée avec la cible (le chef de famille).
    """
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.variables_a_supprimer_ = []
        self.variables_a_conserver_ = []

    def fit(self, X, y):
        # Conversion en DataFrame Pandas si X est un array NumPy
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray) or isinstance(y, pd.Series):
            y_series = pd.Series(y, index=X.index)
        else:
            y_series = y

        # 1. Calcul de la matrice de corrélation
        corr_matrix = X.corr().abs()

        # 2. Identification des paires colinéaires
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                val = corr_matrix.iloc[i, j]
                if val >= self.threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

        # 3. Regroupement par familles (Pure Python)
        familles = []
        for col1, col2 in high_corr_pairs:
            trouve = False
            for f in familles:
                if col1 in f or col2 in f:
                    f.add(col1)
                    f.add(col2)
                    trouve = True
                    break
            if not trouve:
                familles.append({col1, col2})

        # Fusion des sous-groupes interconnectés
        familles_fusions = []
        while familles:
            first, *rest = familles
            first = set(first)
            lf = -1
            while len(first) > lf:
                lf = len(first)
                rest2 = []
                for r in rest:
                    if first & set(r):
                        first |= set(r)
                    else:
                        rest2.append(r)
                rest = rest2
            familles_fusions.append(list(first))
            familles = rest

        # 4. Élection du chef de famille basé sur la corrélation avec y
        # Calcul de la corrélation de chaque variable avec la cible y
        corr_with_target = X.apply(lambda col: col.corr(y_series)).abs()

        variables_a_conserver = []
        variables_a_supprimer = set()

        for famille in familles_fusions:
            chef = max(famille, key=lambda var: corr_with_target[var])
            variables_a_conserver.append(chef)
            for var in famille:
                if var != chef:
                    variables_a_supprimer.add(var)

        self.variables_a_conserver_ = variables_a_conserver
        self.variables_a_supprimer_ = list(variables_a_supprimer)
        return self

    def transform(self, X):
        # Suppression des variables redondantes
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.variables_a_supprimer_, errors='ignore')
        else:
            # Si c'est un array numpy
            df_temp = pd.DataFrame(X)
            return df_temp.drop(columns=self.variables_a_supprimer_, errors='ignore').values


def create_preprocessing_pipeline(reduce_features=True, threshold=0.85):
    """
    Crée et renvoie un Pipeline Scikit-Learn.
    
    Parameters:
    -----------
    reduce_features : bool
        Si True, inclut la sélection de variables redondantes.
    threshold : float
        Seuil de corrélation pour la sélection.
    """
    steps = []
    
    if reduce_features:
        steps.append(('feature_selector', CollinearFeatureSelector(threshold=threshold)))
        
    steps.append(('scaler', StandardScaler()))
    
    return Pipeline(steps)


def prepare_data(df, target_col='diagnosis', test_size=0.20, random_state=42):
    """
    Sépare les données en Train / Test et prépare les deux pipelines :
    - Baseline (30 variables)
    - Réduit (16 variables)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 1. Pipeline Baseline (Juste le StandardScaler)
    pipeline_baseline = create_preprocessing_pipeline(reduce_features=False)
    X_train_base_scaled = pipeline_baseline.fit_transform(X_train)
    X_test_base_scaled = pipeline_baseline.transform(X_test)

    # 2. Pipeline Réduit (Sélection + StandardScaler)
    pipeline_reduced = create_preprocessing_pipeline(reduce_features=True)
    X_train_red_scaled = pipeline_reduced.fit_transform(X_train, y_train)
    X_test_red_scaled = pipeline_reduced.transform(X_test)

    return {
        'baseline': (X_train_base_scaled, X_test_base_scaled),
        'reduced': (X_train_red_scaled, X_test_red_scaled),
        'targets': (y_train, y_test),
        'pipelines': {
            'baseline': pipeline_baseline,
            'reduced': pipeline_reduced
        }
    }