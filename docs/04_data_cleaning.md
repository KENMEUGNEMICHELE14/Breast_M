# Étape 4 : Data Cleaning & Preprocessing

## 1. Contexte & Objectifs
L'objectif de cette étape est de préparer le jeu de données **Breast Cancer Wisconsin** pour l'entraînement des modèles de Machine Learning. Afin d'obtenir des modèles performants et généralisables, nous avons mis en place une chaîne de traitement (*Pipeline*) rigoureuse permettant d'éviter tout **Data Leakage** (fuite de données).

## 2. Découpage du Jeu de Données (Train / Test Split)
Afin de préserver l'impartialité de l'évaluation, le jeu de données initial (30 variables prédictives) a été séparé avant toute transformation :
* **Répartition** : 80% Entraînement (`X_train`, `y_train`) / 20% Test (`X_test`, `y_test`).
* **Stratification** : Option `stratify=y` activée pour conserver la même proportion de classes (tumeurs bénignes et malignes) dans les deux sous-ensembles.
* **Reproductibilité** : `random_state=42`.

## 3. Traitement de la Multicolinéarité (Sélection de Variables)
Les variables géométriques du jeu de données présentent de très fortes corrélations entre elles (ex. rayon, périmètre et surface). Pour éviter la redondance et simplifier le modèle sans perdre d'information :
* **Seuil de corrélation** : $|r| \ge 0{,}85$.
* **Algorithme des « Chefs de famille »** :
  1. Identification des paires de variables fortement corrélées.
  2. Regroupement des variables interconnectées en familles de colinéarité.
  3. Sélection du « chef de famille » : la variable du groupe ayant la plus forte corrélation absolue avec la cible (`diagnosis`).
  4. Suppression des variables redondantes.

### Résultat de la sélection :
* **3 familles principales** identifiées.
* **Chefs de famille conservés** : `concave points_worst`, `radius_se`, `texture_worst`.
* **14 variables redondantes supprimées** (ex. `radius_mean`, `area_mean`, `perimeter_worst`, etc.).
* **Passage de 30 variables (Baseline) à 16 variables clés (Jeu réduit)**.

## 4. Normalisation / Standardisation (`StandardScaler`)
Comme les caractéristiques possèdent des échelles très différentes (ex. `area` ~ 2000 vs `concavity` ~ 0.1), une standardisation $z = \frac{x - \mu}{\sigma}$ a été appliquée :
* **Ajustement (`fit_transform`)** : Réalisé **uniquement sur le jeu d'entraînement** (`X_train`) pour calculer la moyenne ($\mu$) et l'écart-type ($\sigma$).
* **Transformation (`transform`)** : Appliquée au jeu de test (`X_test`) en utilisant **strictement les paramètres du train**, garantissant l'absence de *Data Leakage*.

## 5. Industrialisation : Structuration dans `src/preprocessing.py`
L'ensemble de la logique a été modularisé dans le fichier `src/preprocessing.py` à l'aide des classes et fonctions Scikit-Learn :
* `CollinearFeatureSelector` : Transformateur personnalisé (`BaseEstimator`, `TransformerMixin`).
* `create_preprocessing_pipeline()` : Assemblage de la sélection de variables et du `StandardScaler` dans un `Pipeline` scikit-learn.
* `prepare_data()` : Fonction principale renvoyant les jeux **Baseline** (30 variables) et **Réduit** (16 variables) prêts pour l'entraînement.

---
