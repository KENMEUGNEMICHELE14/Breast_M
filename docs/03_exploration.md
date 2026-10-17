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





# Partie 2 : Nettoyage des Données, Qualité et Traitement des Anomalies (*Data Cleaning*)

Le nettoyage des données constitue une étape cruciale du prétraitement. Un jeu de données mal nettoyé peut générer des biais statistiques majeurs et déstabiliser les futurs modèles d'apprentissage automatique. Dans cette section, nous appliquons une démarche systématique d'identification et de traitement des erreurs.

---

## 1. Gestion des Valeurs Manquantes
* **Détection** : La vérification initiale via `df.isnull().sum()` a mis en évidence la présence d'une colonne parasite issue de l'importation du fichier CSV, intitulée `Unnamed: 32`, qui contenait 569 valeurs manquantes (soit 100 % de nuls).
* **Traitement (Amputation)** : Cette colonne ne véhiculant aucune information utile a été entièrement supprimée (`df.drop(columns=['Unnamed: 32'])`).
* **Résultat** : Après suppression, le jeu de données comporte **0 valeur manquante** sur l'ensemble des 569 observations. Aucune méthode d'imputation (moyenne/médiane) ou d'amputation de lignes n'a été nécessaire.

---

## 2. Harmonisation du Typage des Variables
* **Détection** : La variable cible `diagnosis` était initialement enregistrée sous forme de chaîne de caractères (`object`) avec les modalités `"M"` (Maligne) et `"B"` (Bénigne).
* **Traitement (Encodage binaire)** : Pour permettre son exploitation dans les calculs statistiques et les algorithmes de Machine Learning, la variable a été convertie en entier (`int64`) :
  * `1` : Tumeur maligne
  * `0` : Tumeur bénigne
* **Résultat** : Toutes les variables explicatives du jeu de données sont désormais au format numérique continu (`float64`) et la cible en binaire (`int64`).

---

## 3. Contrôle des Incohérences Physiques et Lexicales
* **Vérification** : Les caractéristiques morphologiques de la cellule (rayon, périmètre, surface, texture, etc.) représentent des mesures physiques réelles.
* **Résultat** : L'analyse des valeurs minimales (`df.min()`) confirme que toutes les grandeurs géométriques majeures sont strictement positives ($> 0$). Les valeurs égales à 0 constatées uniquement sur les variables de concavité reflètent des contours lisses, ce qui est biologiquement cohérent.

---

## 4. Analyse des Outliers : Valeurs Aberrantes vs Valeurs Atypiques
* **Détection** : L'application automatique de la règle statistique de l'Écart Interquartile ($\text{IQR} = Q3 - Q1$) identifie des points statistiques hors normes sur plusieurs caractéristiques (notamment 25 points pour `area_mean`, 38 pour `radius_se` et 65 pour `area_se`).
* **Analyse visuelle et médicale par Boxplot** :
  * La superposition des distributions croisées avec la variable `diagnosis` démontre que la quasi-totalité de ces valeurs extrêmes provient des patientes atteintes de **tumeurs malignes (`1`)**.
  * Ces points mesurent des tumeurs volumineuses, irrégulières ou à forte variabilité cellulaire (stades avancés).
* **Décision métier** : Il s'agit de **valeurs atypiques réelles** (signature biologique de la sévérité de la maladie) et non de valeurs aberrantes (erreurs de mesure ou de saisie). Elles sont donc **conservées intactes** afin d'éviter tout biais d'échantillonnage et de permettre aux modèles d'apprendre à détecter les cas graves.

---

## 5. Détection des Doublons
* **Vérification** : Une recherche globale d'observations strictement identiques sur l'ensemble des caractéristiques (`df.duplicated().sum()`) a été réalisée.
* **Résultat** : **0 doublon** détecté. Chaque ligne représente une observation unique et valide.

---

## Bilan de la Qualité des Données
À l'issue de cette étape de nettoyage, le DataFrame est composé de **569 lignes et 31 colonnes** exploitables, garantissant un niveau de qualité optimal pour aborder l'analyse bivariée et l'étude des corrélations.




# Partie 3 : Analyse des Corrélations et Exploration des Relations (*Correlation Analysis & Heatmap*)

Une fois le jeu de données nettoyé, l'analyse bivariée et l'étude des corrélations permettent de comprendre la structure des données. L'objectif de cette étape est double :
1. **Évaluer la force de l'association entre chaque variable explicative et la variable cible (`diagnosis`)** pour repérer les indicateurs clés de la malignité.
2. **Détecter la multicolinéarité (redondance d'information)** entre les caractéristiques explicatives afin de rationaliser le choix des variables pour la future phase de modélisation.

---

## 1. Méthodologie et Calcul de la Matrice de Corrélation
Nous utilisons le **coefficient de corrélation de Pearson ($r$)**, qui mesure l'intensité de la relation linéaire entre deux variables continuës, avec des valeurs comprises entre $-1$ et $+1$ :
* $r \approx +1$ : Forte corrélation positive (les deux variables augmentent ensemble).
* $r \approx 0$ : Absence de relation linéaire.
* $r \approx -1$ : Forte corrélation négative.

*Remarque méthodologique* : Le coefficient de Pearson étant invariant par transformation linéaire ($r_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$), la mise à l'échelle (*scaling*) n'est pas requise à ce stade et donnerait une matrice rigoureusement identique.

---

## 2. Visualisation Globale : La Carte Thermique (*Heatmap*)
La matrice de corrélation globale (31x31) a été visualisée à l'aide d'une carte thermique (*Heatmap* Seaborn). L'observation de cette carte révèle deux éléments majeurs :
* Une présence marquée de blocs de couleur intense (rouge), traduisant une forte interdépendance entre plusieurs groupes de variables morphologiques.
* Des associations claires et positives entre de nombreuses caractéristiques cellulaires et la sévérité du diagnostic.

---

## 3. Identification des Meilleurs Prédicteurs du Diagnostic
L'extraction des coefficients de corrélation par rapport à la cible `diagnosis` met en évidence les variables les plus discriminantes :

* **Les facteurs prédictifs majeurs ($r > 0,70$)** :
  * `concave_points_worst` ($r \approx 0,79$) et `concave_points_mean` ($r \approx 0,78$)
  * `perimeter_worst` ($r \approx 0,78$) et `radius_worst` ($r \approx 0,78$)
  * `area_worst` ($r \approx 0,73$)
* **Interprétation médicale** : La présence de points de concavité (renfoncements de la membrane cellulaire) ainsi qu'un rayon/périmètre/surface très élevés dans leurs pires valeurs (*worst*) sont les signes physiques les plus corrélés à une tumeur maligne.

---

## 4. Analyse de la Multicolinéarité (Redondance des Variables)
L'analyse croisée des variables explicatives confirme l'existence de très fortes corrélations entre plusieurs d'entre elles ($\vert{}r\vert{} \ge 0,85$), notamment :

* **Trinôme Géométrique (`radius`, `perimeter`, `area`)** :
  * `radius_mean`, `perimeter_mean` et `area_mean` présentent des corrélations proches de $0,98$ à $0,99$.
  * *Explication physique* : Il s'agit d'une redondance mathématique directe ($P \approx 2\pi r$ et $S \approx \pi r^2$). Ces trois métriques décrivent la même propriété fondamentale : la taille globale de la cellule.
* **Corrélations entre catégories d'erreurs et pires valeurs** :
  * Les variables de moyenne (`_mean`) sont très fortement corrélées avec leurs contreparties extrêmes (`_worst`) et d'erreur standard (`_se`).

---


# Partie 4 Hypothèses de Modélisation (Machine Learning)

En nous appuyant sur l'analyse de corrélation, nous posons trois hypothèses pour la phase d'apprentissage automatique :

* **Hypothèse 1 (Prédicteurs principaux — $r \ge 0{,}70$)** : Les variables mesurant la concavité et les dimensions maximales de la cellule (`concave points_worst`, `perimeter_worst`, `radius_worst`, `area_worst`) seront les éléments les plus importants pour distinguer une tumeur maligne d'une tumeur bénigne.
* **Hypothèse 2 (Sensibilité à la redondance — $r \ge 0{,}85$)** : La forte corrélation entre le rayon, le périmètre et la surface ($r > 0{,}98$) perturbera un modèle linéaire comme la **Régression Logistique**, alors qu'un modèle basé sur des arbres comme **Random Forest** la supportera très bien.
* **Hypothèse 3 (Simplification sans perte de qualité)** : Supprimer les variables redondantes ($\vert{}r\vert{} \ge 0{,}85$) permettra de simplifier le jeu de données tout en conservant un **Rappel (*Recall*)** maximal sur les cas maligns, ce qui est indispensable sur le plan médical pour ne rater aucun cancer.