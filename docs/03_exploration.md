# Partie 1 : Description et Distribution des Données (EDA)

## 1. Structure générale du jeu de données
* **Volume initial** : 569 observations (patients) et 33 variables.
* **Prétraitement de la variable cible** : La colonne `diagnosis` a été convertie en valeurs numériques (`M` = 1 pour maligne, `B` = 0 pour bénigne).
* **Nettoyage initial** : La colonne `id` a été supprimée du tableau car elle ne constitue pas une caractéristique médicale informative.
* **Types de données** : Toutes les caractéristiques explicatives sont de type continu (`float64`), et la cible est entière (`int64`).

## 2. Statistiques Descriptives
* **Répartition de la cible (`diagnosis`)** : La moyenne de la variable cible est de `0.37`, ce qui indique une proportion d'environ **37.3% de tumeurs malignes** et **62.7% de tumeurs bénignes** dans le jeu de données.
* **Écarts d'échelle** : Les variables présentent des ordres de grandeur très différents (par exemple, `area_mean` a une moyenne autour de 654 alors que `smoothness_mean` a une moyenne autour de 0.09).
* **Implication pour la modélisation** : Une étape de **standardisation ou normalisation** sera indispensable avant l'entraînement des modèles de Machine Learning.

## 3. Analyse des Distributions et Asymétrie (Skewness)
* **Distributions symétriques** : Certaines variables (comme `smoothness_mean` ou `symmetry_mean`) suivent une distribution quasi-symétrique (en forme de cloche).
* **Distributions asymétriques** : Plusieurs variables affichent un coefficient d'asymétrie (*skewness*) élevé (> 1), notamment `area_mean` et `concavity_mean`. Leurs histogrammes sont étirés vers la droite, indiquant la présence de quelques tumeurs avec des mesures extrêmement fortes.