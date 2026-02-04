# Rapport de Projet - Qualité des Données

## Découverte de Dépendances Fonctionnelles Significatives : Approche Hybride combinant Algorithmes Classiques et Large Language Models

---

**Master 2 IASD - Université Paris-Dauphine**

**Cours : Qualité des Données**

**Date : Février 2026**

---

# Table des Matières

1. [Introduction](#1-introduction)
2. [Présentation des Datasets](#2-présentation-des-datasets)
3. [Task 1 : Découverte Algorithmique des FDs](#3-task-1--découverte-algorithmique-des-fds)
4. [Task 2 : Analyse Sémantique avec LLM](#4-task-2--analyse-sémantique-avec-llm)
5. [Task 3 : Échantillonnage et Hypothèses FD](#5-task-3--échantillonnage-et-hypothèses-fd)
6. [Task 4 : Pipeline Hybride Algorithme + LLM](#6-task-4--pipeline-hybride-algorithme--llm)
7. [Conclusion Générale](#7-conclusion-générale)
8. [Annexes](#8-annexes)

---

# 1. Introduction

## 1.1 Contexte et Motivation

Dans le domaine des bases de données et de la qualité des données, les **dépendances fonctionnelles (FDs)** jouent un rôle fondamental. Une dépendance fonctionnelle `X → Y` exprime une contrainte selon laquelle, pour toute paire de tuples ayant les mêmes valeurs pour les attributs X (le déterminant ou LHS - Left-Hand Side), ils doivent nécessairement avoir les mêmes valeurs pour l'attribut Y (le déterminé ou RHS - Right-Hand Side).

**Exemple concret :** Dans une table d'employés, `NuméroEmployé → Nom` signifie que chaque numéro d'employé détermine de manière unique le nom de l'employé.

## 1.2 Problématique

Les algorithmes classiques de découverte de FDs (TANE, FUN, FD_Mine, FastFDs) sont capables de découvrir **toutes** les dépendances fonctionnelles valides dans un dataset. Cependant, cette exhaustivité pose un problème majeur : parmi toutes ces FDs découvertes, beaucoup sont :

- **Triviales** : FDs issues de clés primaires (ex: `ID → tout`)
- **Accidentelles** : Corrélations statistiques sans lien causal
- **Non significatives** : Pas de valeur métier ou sémantique
- **Sur-ajustées** : Vraies uniquement par hasard sur les données actuelles

## 1.3 Objectif du Projet

Ce projet vise à développer une **approche hybride** combinant :
1. **Algorithmes classiques** : Pour la découverte exhaustive et précise des FDs
2. **Large Language Models (LLMs)** : Pour l'évaluation sémantique et le filtrage intelligent

L'objectif final est d'identifier les FDs **significatives** - celles qui ont un sens métier réel et une valeur pour la compréhension des données.

## 1.4 Méthodologie Générale

Le projet se décompose en 4 tâches complémentaires :

| Task | Objectif | Approche |
|------|----------|----------|
| Task 1 | Découverte algorithmique | Algorithmes classiques |
| Task 2 | Analyse sémantique | Évaluation par LLM |
| Task 3 | Étude de l'échantillonnage | Validation hypothèses vs réalité |
| Task 4 | Pipeline hybride | Combinaison des approches |

---

# 2. Présentation des Datasets

## 2.1 Vue d'ensemble

Nous avons utilisé 5 datasets provenant du UCI Machine Learning Repository, choisis pour leur diversité de domaines et de structures :

| Dataset | Domaine | Lignes | Colonnes | Type de données |
|---------|---------|--------|----------|-----------------|
| IRIS | Botanique | 150 | 5 | Numériques continues + 1 catégorielle |
| BRIDGES | Génie civil | 108 | 13 | Mixte (ID, catégorielles, numériques) |
| ABALONE | Biologie marine | 4177 | 9 | Numériques continues + 1 catégorielle |
| BREAST-CANCER | Médecine | 699 | 11 | Numériques discrètes |
| NURSERY | Social | 12960 | 9 | Toutes catégorielles |

## 2.2 Description détaillée des datasets

### 2.2.1 IRIS (Fleurs d'Iris)

**Contexte :** Classification de 3 espèces de fleurs d'iris basée sur les mesures des sépales et pétales.

| Attribut | Type | Description |
|----------|------|-------------|
| sepal_length | Float | Longueur du sépale (cm) |
| sepal_width | Float | Largeur du sépale (cm) |
| petal_length | Float | Longueur du pétale (cm) |
| petal_width | Float | Largeur du pétale (cm) |
| class | Catégoriel | Espèce (setosa, versicolor, virginica) |

**Caractéristiques :** Dataset classique de machine learning, bien équilibré (50 instances par classe).

### 2.2.2 BRIDGES (Ponts de Pittsburgh)

**Contexte :** Caractéristiques des ponts de la région de Pittsburgh.

| Attribut | Type | Description |
|----------|------|-------------|
| IDENTIF | String | Identifiant unique du pont |
| RIVER | Catégoriel | Rivière traversée |
| LOCATION | Numérique | Localisation |
| ERECTED | Numérique | Année de construction |
| PURPOSE | Catégoriel | Usage (route, train, piéton) |
| LENGTH | Numérique | Longueur |
| LANES | Numérique | Nombre de voies |
| CLEAR-G | Catégoriel | Dégagement vertical |
| T-OR-D | Catégoriel | Type de tablier |
| MATERIAL | Catégoriel | Matériau (acier, fer, bois) |
| SPAN | Catégoriel | Type de portée |
| REL-L | Catégoriel | Longueur relative |
| TYPE | Catégoriel | Type de pont |

**Caractéristiques :** Présence d'une clé primaire (IDENTIF), valeurs manquantes représentées par "?".

### 2.2.3 ABALONE (Ormeaux)

**Contexte :** Prédiction de l'âge des ormeaux à partir de mesures physiques.

| Attribut | Type | Description |
|----------|------|-------------|
| Sex | Catégoriel | Sexe (M, F, I=infant) |
| Length | Float | Longueur de la coquille |
| Diameter | Float | Diamètre |
| Height | Float | Hauteur |
| Whole_weight | Float | Poids total |
| Shucked_weight | Float | Poids de la chair |
| Viscera_weight | Float | Poids des viscères |
| Shell_weight | Float | Poids de la coquille |
| Rings | Integer | Nombre d'anneaux (≈ âge) |

**Caractéristiques :** Grand dataset avec attributs fortement corrélés.

### 2.2.4 BREAST-CANCER (Cancer du sein)

**Contexte :** Diagnostic de tumeurs mammaires (bénignes vs malignes).

| Attribut | Type | Description |
|----------|------|-------------|
| id | Integer | Identifiant du patient |
| clump_thickness | Integer (1-10) | Épaisseur de l'amas |
| uniformity_cell_size | Integer (1-10) | Uniformité taille cellules |
| uniformity_cell_shape | Integer (1-10) | Uniformité forme cellules |
| marginal_adhesion | Integer (1-10) | Adhésion marginale |
| single_epithelial_cell_size | Integer (1-10) | Taille cellule épithéliale |
| bare_nuclei | Integer (1-10) | Noyaux nus |
| bland_chromatin | Integer (1-10) | Chromatine fade |
| normal_nucleoli | Integer (1-10) | Nucléoles normaux |
| mitoses | Integer (1-10) | Mitoses |
| class | Integer | Diagnostic (2=bénin, 4=malin) |

**Caractéristiques :** Données médicales avec identifiant unique, quelques valeurs manquantes.

### 2.2.5 NURSERY (Évaluation de crèches)

**Contexte :** Évaluation de demandes d'admission en crèche.

| Attribut | Type | Description |
|----------|------|-------------|
| parents | Catégoriel | Occupation des parents |
| has_nurs | Catégoriel | Garde d'enfants actuelle |
| form | Catégoriel | Structure familiale |
| children | Catégoriel | Nombre d'enfants |
| housing | Catégoriel | Conditions de logement |
| finance | Catégoriel | Situation financière |
| social | Catégoriel | Conditions sociales |
| health | Catégoriel | Santé de l'enfant |
| class | Catégoriel | Décision (not_recom, recommend, etc.) |

**Caractéristiques :** Grand dataset entièrement catégoriel, conçu pour la classification.

---

# 3. Task 1 : Découverte Algorithmique des FDs

## 3.1 Objectif

Implémenter un algorithme de découverte de dépendances fonctionnelles et l'appliquer sur les datasets pour obtenir une liste exhaustive des FDs valides.

## 3.2 Algorithme Implémenté

### 3.2.1 Principe

L'algorithme implémenté teste systématiquement toutes les combinaisons possibles :

```
Pour chaque taille de LHS de 1 à max_lhs_size:
    Pour chaque combinaison de colonnes de cette taille:
        Pour chaque colonne RHS possible:
            Si RHS n'est pas dans LHS:
                Calculer la validité de la FD
                Si validité >= seuil:
                    Ajouter la FD à la liste
```

### 3.2.2 Calcul de la validité

Pour une FD `X → Y`, la validité est calculée comme suit :

1. Grouper les tuples par les valeurs de X
2. Pour chaque groupe, compter le nombre de valeurs distinctes de Y
3. Une violation existe si un groupe a plus d'une valeur de Y
4. Validité = (Groupes sans violation / Total groupes) × 100%

### 3.2.3 Code de la fonction principale

```python
def check_fd(df, lhs_cols, rhs_col):
    grouped = df.groupby(list(lhs_cols))[rhs_col].nunique()
    violations = (grouped > 1).sum()
    total_groups = len(grouped)
    validity_rate = ((total_groups - violations) / total_groups) * 100
    holds = violations == 0
    return holds, validity_rate, violations
```

## 3.3 Résultats de la découverte

### 3.3.1 FDs exactes (100% de validité)

| Dataset | Nombre de FDs exactes | Observation |
|---------|----------------------|-------------|
| IRIS | 0 | Aucune FD parfaite (données continues) |
| BRIDGES | 171 | Nombreuses FDs dues à la clé primaire IDENTIF |
| ABALONE | 0 | Aucune FD parfaite |
| BREAST-CANCER | ~50 | FDs liées à l'identifiant |
| NURSERY | 0 | Aucune FD parfaite |

### 3.3.2 FDs approximatives (≥ 90% de validité)

| Dataset | FDs ≥ 90% | Exemples |
|---------|-----------|----------|
| IRIS | 6 | `petal_length, petal_width -> class` (99%) |
| BRIDGES | 325 | `IDENTIF -> *` (100%), `LOCATION -> RIVER` (100%) |
| ABALONE | Peu | Corrélations partielles entre mesures |
| NURSERY | 0 | Données trop variées |

### 3.3.3 Détail des FDs découvertes pour IRIS

| FD | Validité | Violations |
|----|----------|------------|
| sepal_length, petal_length -> class | 99.19% | 1 |
| petal_length, petal_width -> class | 99.02% | 1 |
| sepal_width, petal_length -> class | 98.36% | 2 |
| sepal_width, petal_width -> class | 96.91% | 3 |
| sepal_length, petal_width -> class | 96.36% | 4 |
| sepal_length, sepal_width -> class | 91.38% | 5 |

### 3.3.4 Détail des FDs découvertes pour BRIDGES

| FD | Validité | Type |
|----|----------|------|
| IDENTIF -> RIVER | 100% | Clé primaire |
| IDENTIF -> LOCATION | 100% | Clé primaire |
| IDENTIF -> ERECTED | 100% | Clé primaire |
| IDENTIF -> PURPOSE | 100% | Clé primaire |
| IDENTIF -> LENGTH | 100% | Clé primaire |
| IDENTIF -> LANES | 100% | Clé primaire |
| IDENTIF -> MATERIAL | 100% | Clé primaire |
| IDENTIF -> TYPE | 100% | Clé primaire |
| LOCATION -> RIVER | 100% | Règle géographique |

## 3.4 Analyse des résultats

### 3.4.1 Observations principales

1. **Datasets avec clé primaire (BRIDGES, BREAST-CANCER)** : Génèrent de nombreuses FDs triviales de type `ID -> attribut`. Ces FDs sont techniquement vraies mais n'apportent pas d'information sémantique.

2. **Datasets à valeurs continues (IRIS, ABALONE)** : Peu ou pas de FDs exactes car les valeurs numériques continues ont une haute cardinalité.

3. **Datasets catégoriels complexes (NURSERY)** : La combinatoire des valeurs catégorielles empêche l'émergence de FDs même approximatives.

### 3.4.2 Limites de l'approche purement algorithmique

| Limite | Description | Exemple |
|--------|-------------|---------|
| **Trivialité** | Détecte les FDs de clé primaire sans les distinguer | `IDENTIF -> RIVER` |
| **Pas de sémantique** | Impossible de savoir si une FD a un sens métier | `Sex -> Rings` (fausse corrélation) |
| **Volume** | Peut générer des centaines de FDs à analyser manuellement | 325 FDs pour BRIDGES |

## 3.5 Conclusion Task 1

> **L'algorithme de découverte de FDs est efficace pour trouver TOUTES les FDs valides techniquement, mais il ne distingue pas les FDs significatives des FDs triviales ou accidentelles. Une analyse sémantique complémentaire est nécessaire.**

---

# 4. Task 2 : Analyse Sémantique avec LLM

## 4.1 Objectif

Utiliser un Large Language Model (Claude 3 Haiku d'Anthropic) pour évaluer la **signification sémantique** des dépendances fonctionnelles découvertes par les algorithmes.

## 4.2 Méthodologie

### 4.2.1 Configuration du LLM

- **Modèle** : Claude 3 Haiku (claude-3-haiku-20240307)
- **API** : Anthropic API
- **Paramètres** : max_tokens=1024, temperature par défaut

### 4.2.2 Prompt utilisé

Pour chaque FD, le prompt suivant est envoyé au LLM :

```
Tu es un expert en bases de données. Voici un échantillon de données du dataset "{dataset_name}".

Colonnes: {columns}
Échantillon: {sample_data}

En analysant CET ÉCHANTILLON UNIQUEMENT, quelles dépendances fonctionnelles (FDs) semblent tenir ?

Une FD X → Y signifie : si deux lignes ont la même valeur pour X, elles ont la même valeur pour Y.

Liste exactement 5 FDs que tu penses vraies dans cet échantillon.
```

### 4.2.3 Catégories d'évaluation

Le LLM classe chaque FD dans l'une des catégories suivantes :

| Catégorie | Description | Score typique |
|-----------|-------------|---------------|
| **key** | Clé primaire ou identifiant unique | 8-10 |
| **business_rule** | Règle métier logique et significative | 7-10 |
| **derived** | Attribut calculé ou dérivé d'autres | 4-7 |
| **accidental** | Corrélation sans lien causal | 1-4 |
| **meaningless** | Aucun sens sémantique | 0-2 |

## 4.3 Résultats de l'analyse sémantique

### 4.3.1 Exemple d'analyse pour IRIS

| FD | Score LLM | Catégorie | Explication du LLM |
|----|-----------|-----------|-------------------|
| petal_length, petal_width -> class | 9/10 | business_rule | "Pétales déterminant clairement l'espèce de fleur" |
| sepal_length, petal_length -> class | 8/10 | business_rule | "Pétales et sépales définissent les espèces" |
| sepal_width, petal_length -> class | 4/10 | derived | "Petal length and sepal width do not uniquely identify class" |
| sepal_length, sepal_width -> class | 4/10 | derived | "Sepals provide partial information about class" |

**Analyse :** Le LLM identifie correctement que les dimensions des **pétales** sont plus discriminantes que celles des sépales pour la classification des iris - ce qui correspond à la réalité biologique.

### 4.3.2 Exemple d'analyse pour BRIDGES

| FD | Score LLM | Catégorie | Explication du LLM |
|----|-----------|-----------|-------------------|
| IDENTIF -> RIVER | 10/10 | key | "IDENTIF is a unique identifier for each bridge" |
| IDENTIF -> MATERIAL | 9/10 | key | "IDENTIF determines all attributes as primary key" |
| IDENTIF -> PURPOSE | 9/10 | business_rule | "Le but d'un pont est une règle métier essentielle" |
| LOCATION -> RIVER | 8/10 | business_rule | "La localisation détermine la rivière traversée" |

**Analyse :** Le LLM reconnaît IDENTIF comme clé primaire et identifie également `LOCATION -> RIVER` comme une règle géographique logique.

### 4.3.3 Exemple d'analyse pour ABALONE

| FD proposée par LLM | Validité réelle | Catégorie | Commentaire |
|--------------------|-----------------|-----------|-------------|
| Sex -> Rings | FAUSSE (0%) | accidental | Le sexe ne détermine pas l'âge |
| Length, Diameter -> Whole_weight | Partielle (~80%) | derived | Relation physique mais pas exacte |

**Analyse :** Le LLM propose des FDs qui semblent logiques mais ne tiennent pas sur le dataset complet - démontrant les limites de l'analyse sur échantillon.

### 4.3.4 Types de FDs sans signification identifiés

Le LLM a identifié plusieurs types de FDs **non significatives** :

| Type | Description | Exemple |
|------|-------------|---------|
| **Accidentelle** | Corrélation statistique sans causalité | `Sex -> Rings` |
| **Dégénérée** | FD qui tient car une colonne a peu de valeurs | `constant_col -> anything` |
| **Sur-ajustée** | Vraie uniquement sur l'échantillon observé | FDs avec LHS très large |
| **Encodage** | Due à un schéma d'encodage artificiel | `code -> description` |

## 4.4 Comparaison Algorithme vs LLM

### 4.4.1 Forces et faiblesses

| Aspect | Algorithme | LLM |
|--------|------------|-----|
| **Précision technique** | ✅ Exacte | ❌ Peut se tromper |
| **Exhaustivité** | ✅ Trouve tout | ❌ Limité par le prompt |
| **Compréhension sémantique** | ❌ Aucune | ✅ Comprend le contexte |
| **Distinction trivial/significatif** | ❌ Non | ✅ Oui |
| **Scalabilité** | ✅ Rapide | ❌ Coût API |

### 4.4.2 Complémentarité

L'algorithme et le LLM sont **complémentaires** :
- L'algorithme garantit la validité technique
- Le LLM filtre par la pertinence sémantique

## 4.5 Conclusion Task 2

> **Le LLM apporte une dimension sémantique essentielle à l'analyse des FDs. Il peut distinguer les clés primaires, les règles métier significatives, et les corrélations accidentelles. Cependant, il doit être utilisé en complément (et non en remplacement) des algorithmes classiques pour garantir la précision technique.**

---

# 5. Task 3 : Échantillonnage et Hypothèses FD

## 5.1 Objectif

Étudier comment les dépendances fonctionnelles suggérées par un LLM sur des **échantillons** peuvent différer de celles qui tiennent sur le **dataset complet**. L'objectif est d'identifier les **faux positifs** - des FDs qui semblent vraies sur un échantillon mais sont fausses en réalité.

## 5.2 Méthodologie

### 5.2.1 Types d'échantillonnage

Nous avons testé deux stratégies d'échantillonnage :

| Type | Méthode | Avantage | Inconvénient |
|------|---------|----------|--------------|
| **Aléatoire** | Sélection uniforme au hasard | Simple, rapide | Peut manquer des groupes rares |
| **Stratifié** | Préserve les proportions par classe | Représentatif | Plus complexe |

### 5.2.2 Tailles d'échantillons

| Dataset | Taille originale | Échantillon | Ratio |
|---------|------------------|-------------|-------|
| IRIS | 150 | 30 | 20% |
| BRIDGES | 108 | 40 | 37% |
| ABALONE | 4177 | 50 | 1.2% |
| BREAST-CANCER | 699 | 50 | 7% |

### 5.2.3 Processus d'évaluation

1. **Création des échantillons** (aléatoire et stratifié)
2. **Suggestion de FDs par le LLM** sur chaque échantillon
3. **Validation sur l'échantillon** (vérification algorithmique)
4. **Validation sur le dataset complet**
5. **Classification** : Vrai positif, Faux positif, Faux négatif

## 5.3 Résultats détaillés

### 5.3.1 Statistiques globales

| Métrique | Valeur | Pourcentage |
|----------|--------|-------------|
| **Total FDs analysées** | 33 | 100% |
| **Vraies positives** | 10 | 30.3% |
| **Faux positifs** | 3 | **9.1%** |
| **Fausses (détectées comme telles)** | 20 | 60.6% |

### 5.3.2 Les 3 Faux Positifs identifiés

Ces FDs semblaient **100% vraies sur l'échantillon** mais sont **fausses sur le dataset complet** :

| FD | Dataset | Type échantillon | Validité échantillon | Validité complet | Violations cachées |
|----|---------|------------------|---------------------|------------------|-------------------|
| sepal_length, sepal_width -> petal_length | IRIS | random | **100%** | 81.0% | 22 |
| petal_length, petal_width -> class | IRIS | random | **100%** | 99.0% | 1 |
| petal_width -> class | IRIS | stratified | **100%** | 77.3% | 5 |

### 5.3.3 Analyse détaillée du Faux Positif #1

**FD : `sepal_length, sepal_width -> petal_length`**

| Métrique | Échantillon (30 lignes) | Complet (150 lignes) |
|----------|------------------------|---------------------|
| Validité | 100% | 81.0% |
| Groupes testés | 28 | 116 |
| Violations | 0 | 22 |

**Explication :** Sur 30 lignes tirées au hasard, chaque combinaison unique de (sepal_length, sepal_width) correspondait à une seule valeur de petal_length. Mais sur le dataset complet, 22 groupes ont des valeurs de petal_length différentes pour les mêmes valeurs de sépales.

### 5.3.4 Analyse détaillée du Faux Positif #2

**FD : `petal_length, petal_width -> class`**

| Métrique | Échantillon (30 lignes) | Complet (150 lignes) |
|----------|------------------------|---------------------|
| Validité | 100% | 99.0% |
| Violations | 0 | 1 |

**Explication :** Cette FD est **presque vraie** (99% de validité) mais pas exacte. L'échantillon n'a pas capturé la seule exception qui existe dans les données complètes. C'est un cas particulièrement trompeur.

### 5.3.5 Analyse détaillée du Faux Positif #3

**FD : `petal_width -> class`**

| Métrique | Échantillon (30 lignes) | Complet (150 lignes) |
|----------|------------------------|---------------------|
| Validité | 100% | 77.3% |
| Violations | 0 | 5 |

**Explication :** Avec seulement 30 lignes stratifiées, chaque valeur de petal_width observée correspondait à une seule classe. Mais le dataset complet révèle que plusieurs valeurs de petal_width apparaissent dans plusieurs classes.

### 5.3.6 Résultats par dataset

**IRIS :**

| FD | Type | Échantillon | Complet | Catégorie |
|----|------|-------------|---------|-----------|
| petal_length -> petal_width | random | False | False | false_negative |
| sepal_length, sepal_width -> petal_length | random | **True** | **False** | **FALSE_POSITIVE** |
| petal_length, petal_width -> class | random | **True** | **False** | **FALSE_POSITIVE** |
| sepal_length, sepal_width, petal_length, petal_width -> class | random | True | True | true_positive |
| petal_width -> class | stratified | **True** | **False** | **FALSE_POSITIVE** |

**BRIDGES :**

| FD | Type | Échantillon | Complet | Catégorie |
|----|------|-------------|---------|-----------|
| IDENTIF -> RIVER | stratified | True | True | true_positive |
| IDENTIF -> LOCATION | stratified | True | True | true_positive |
| IDENTIF -> ERECTED | stratified | True | True | true_positive |
| IDENTIF -> MATERIAL | stratified | True | True | true_positive |
| LOCATION -> RIVER | random | True | True | true_positive |

**ABALONE :**

| FD | Type | Échantillon | Complet | Catégorie |
|----|------|-------------|---------|-----------|
| Sex -> Rings | random | False | False | false_negative |
| Length, Diameter, Height -> Whole_weight | random | False | False | false_negative |
| Diameter -> Length | stratified | False | False | false_negative |

### 5.3.7 Comparaison aléatoire vs stratifié

| Métrique | Aléatoire | Stratifié |
|----------|-----------|-----------|
| Faux positifs | 2 | 1 |
| Vraies positives | 4 | 6 |
| Taux de faux positifs | 11% | 7% |

**Observation :** L'échantillonnage stratifié produit légèrement moins de faux positifs car il préserve la diversité des classes.

## 5.4 Analyse des causes des faux positifs

### 5.4.1 Facteurs contribuant aux faux positifs

| Facteur | Impact | Exemple |
|---------|--------|---------|
| **Taille d'échantillon** | Petit échantillon → plus de faux positifs | 30 lignes sur 150 |
| **Distribution des données** | Données non uniformes → biais | Groupes rares non échantillonnés |
| **Cardinalité des attributs** | Haute cardinalité → FDs accidentelles | Valeurs continues |
| **Corrélations partielles** | FDs "presque vraies" → trompeur | 99% de validité |

### 5.4.2 Pourquoi l'échantillonnage crée des faux positifs

```
Dataset complet (150 lignes):
   Groupe A: valeurs {1, 2, 3} pour Y  → Violation!
   Groupe B: valeurs {4} pour Y       → OK
   Groupe C: valeurs {5, 6} pour Y    → Violation!

Échantillon (30 lignes):
   Groupe A: valeurs {1} pour Y       → OK (par hasard)
   Groupe B: valeurs {4} pour Y       → OK
   Groupe C: non échantillonné        → Ignoré

Résultat: La FD semble vraie sur l'échantillon!
```

## 5.5 Implications pratiques

### 5.5.1 Risques des faux positifs

| Risque | Conséquence |
|--------|-------------|
| **Contraintes incorrectes** | Rejet de données valides |
| **Normalisation erronée** | Schéma de base incorrect |
| **Règles métier fausses** | Décisions basées sur des corrélations inexistantes |

### 5.5.2 Recommandations

1. **Ne jamais valider des FDs uniquement sur un échantillon**
2. **Utiliser des échantillons stratifiés** plutôt qu'aléatoires
3. **Augmenter la taille de l'échantillon** quand possible
4. **Toujours vérifier sur le dataset complet** avant d'adopter une FD

## 5.6 Conclusion Task 3

> **9% des FDs suggérées par le LLM sur des échantillons étaient des FAUX POSITIFS - vraies sur l'échantillon mais fausses sur le dataset complet. L'échantillonnage crée des HYPOTHÈSES qui doivent impérativement être validées algorithmiquement sur le dataset complet avant d'être considérées comme des contraintes.**

---

# 6. Task 4 : Pipeline Hybride Algorithme + LLM

## 6.1 Objectif

Développer un **pipeline hybride** qui combine :
- La **rigueur algorithmique** pour la découverte et validation des FDs
- L'**intelligence sémantique du LLM** pour l'évaluation de la pertinence

## 6.2 Architecture du Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE HYBRIDE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Dataset ──► [Algorithme de découverte] ──► FDs candidates         │
│                        │                           │                 │
│                        │ Validité ≥ 90%           │                 │
│                        ▼                           ▼                 │
│              FDs techniquement      ──►  [LLM Évaluation]           │
│                  valides                        │                    │
│                                                 │ Score 0-10        │
│                                                 │ Catégorie         │
│                                                 ▼                    │
│                                    ┌─────────────────────┐          │
│                                    │   Score Hybride     │          │
│                                    │ = (Tech + Sem) / 2  │          │
│                                    └─────────────────────┘          │
│                                                 │                    │
│                    ┌────────────────────────────┼────────────────┐  │
│                    ▼                            ▼                ▼  │
│            Score ≥ 7               5 ≤ Score < 7         Score < 5  │
│         SIGNIFICATIVE               UTILE              À IGNORER    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 6.3 Composants du Pipeline

### 6.3.1 Étape 1 : Découverte algorithmique

**Paramètres :**
- Seuil de validité : ≥ 90%
- Taille max du LHS : 2 attributs
- FDs approximatives incluses

**Sortie :** Liste de FDs candidates avec leur taux de validité

### 6.3.2 Étape 2 : Évaluation sémantique par LLM

**Prompt utilisé :**
```
Tu es un expert en bases de données et en qualité des données.

Dataset: {dataset_name}
Colonnes disponibles: {columns}

Évalue cette dépendance fonctionnelle: {fd}

Réponds UNIQUEMENT avec ce format JSON:
{
    "score": <nombre de 0 à 10>,
    "category": "<key|business_rule|derived|accidental|meaningless>",
    "reason": "<explication courte>"
}
```

**Sortie :** Score (0-10), catégorie, explication

### 6.3.3 Étape 3 : Calcul du score hybride

```
Score_hybride = (Validité_technique/10 + Score_sémantique) / 2
```

| Composante | Plage | Contribution |
|------------|-------|--------------|
| Validité technique | 90-100% → 9-10 | 50% |
| Score sémantique | 0-10 | 50% |
| **Score hybride** | **0-10** | **100%** |

### 6.3.4 Étape 4 : Classification finale

| Score hybride | Classification | Action recommandée |
|---------------|----------------|-------------------|
| ≥ 7 | **SIGNIFICATIVE** | À intégrer dans le schéma |
| 5 - 7 | **UTILE** | À considérer selon le contexte |
| < 5 | **À IGNORER** | Pas de valeur ajoutée |

## 6.4 Résultats du Pipeline Hybride

### 6.4.1 Vue d'ensemble

| Métrique | Valeur |
|----------|--------|
| Datasets analysés | 3 (IRIS, BRIDGES, NURSERY) |
| FDs candidates | 16 |
| FDs significatives (≥7) | **12 (75%)** |
| FDs utiles (5-7) | 4 (25%) |
| FDs à ignorer (<5) | 0 (0%) |

### 6.4.2 Résultats détaillés par FD

| Rang | Dataset | FD | Validité Tech. | Score Sém. | **Score Hybride** | Catégorie |
|------|---------|-----|----------------|------------|-------------------|-----------|
| 1 | BRIDGES | IDENTIF -> RIVER | 100% | 10 | **10.0** | key |
| 2 | BRIDGES | IDENTIF -> LOCATION | 100% | 9 | **9.5** | key |
| 3 | BRIDGES | IDENTIF -> SPAN | 100% | 9 | **9.5** | key |
| 4 | BRIDGES | IDENTIF -> T-OR-D | 100% | 9 | **9.5** | key |
| 5 | BRIDGES | IDENTIF -> LANES | 100% | 9 | **9.5** | key |
| 6 | BRIDGES | IDENTIF -> LENGTH | 100% | 9 | **9.5** | key |
| 7 | BRIDGES | IDENTIF -> PURPOSE | 100% | 9 | **9.5** | business_rule |
| 8 | BRIDGES | IDENTIF -> ERECTED | 100% | 9 | **9.5** | key |
| 9 | IRIS | petal_length, petal_width -> class | 99% | 9 | **9.5** | business_rule |
| 10 | BRIDGES | IDENTIF -> CLEAR-G | 100% | 8 | **9.0** | key |
| 11 | BRIDGES | IDENTIF -> MATERIAL | 100% | 8 | **9.0** | key |
| 12 | IRIS | sepal_length, petal_length -> class | 99% | 8 | **9.0** | business_rule |
| 13 | IRIS | sepal_width, petal_length -> class | 98% | 4 | 6.9 | derived |
| 14 | IRIS | sepal_width, petal_width -> class | 97% | 4 | 6.8 | derived |
| 15 | IRIS | sepal_length, sepal_width -> class | 91% | 4 | 6.6 | derived |
| 16 | IRIS | sepal_length, petal_width -> class | 96% | 3 | 6.3 | derived |

### 6.4.3 Statistiques par catégorie

| Catégorie | Nombre | Score moyen | Interprétation |
|-----------|--------|-------------|----------------|
| **key** | 9 | 9.44 | Clés primaires - très significatives |
| **business_rule** | 3 | 9.30 | Règles métier - très significatives |
| **derived** | 4 | 6.66 | Attributs dérivés - utiles mais secondaires |

### 6.4.4 Distribution des scores

```
Score Hybride    Nombre de FDs    Pourcentage
────────────────────────────────────────────
9.0 - 10.0       │████████████│   12 FDs (75%)
7.0 - 8.9        │            │    0 FDs (0%)
5.0 - 6.9        │████        │    4 FDs (25%)
< 5.0            │            │    0 FDs (0%)
```

## 6.5 Analyse des résultats

### 6.5.1 FDs significatives identifiées

**Catégorie "key" (9 FDs) :**
- Toutes les FDs de type `IDENTIF -> attribut` dans BRIDGES
- Le LLM reconnaît correctement IDENTIF comme clé primaire
- Score technique parfait (100%) + score sémantique élevé (8-10)

**Catégorie "business_rule" (3 FDs) :**
- `petal_length, petal_width -> class` : Les dimensions des pétales déterminent l'espèce (sens biologique)
- `sepal_length, petal_length -> class` : Combinaison de mesures déterminante
- `IDENTIF -> PURPOSE` : Le but d'un pont est une caractéristique métier

### 6.5.2 FDs utiles mais secondaires

**Catégorie "derived" (4 FDs) :**
- `sepal_width, petal_length -> class` : Score sémantique faible (4/10)
- Le LLM explique : "do not uniquely identify class" - information partielle

**Pourquoi ces FDs sont "derived" :**
- Validité technique élevée (91-98%)
- Mais le LLM comprend qu'elles sont moins directes/causales
- Utiles pour la prédiction mais pas des contraintes fondamentales

### 6.5.3 Comparaison avec une approche non-hybride

| Approche | FDs retenues | Problèmes |
|----------|--------------|-----------|
| **Algorithme seul (≥90%)** | 325 (BRIDGES) + 6 (IRIS) = 331 | Trop nombreuses, beaucoup triviales |
| **LLM seul** | Variable, non fiable | Peut halluciner, pas de validation |
| **Hybride** | **16 candidates → 12 significatives** | Optimal |

## 6.6 Visualisations

### 6.6.1 Score Technique vs Sémantique

```
Score Sémantique
     10 │    ●●●●●●●●●              (key)
        │    ●●                      (key)
      8 │    ●●                      (business_rule)
        │
      6 │
        │
      4 │              ○○○○         (derived)
        │
      2 │
        │
      0 └─────────────────────────────
        90%   92%   94%   96%   98%   100%
                  Score Technique

● = FD significative (score ≥ 7)
○ = FD utile (score 5-7)
```

### 6.6.2 Répartition par catégorie

```
        Répartition des FDs

   key ████████████████████ 56.25% (9 FDs)

   business_rule ██████ 18.75% (3 FDs)

   derived ████████ 25% (4 FDs)

   accidental/meaningless: 0%
```

## 6.7 Avantages du Pipeline Hybride

| Avantage | Description |
|----------|-------------|
| **Précision** | Algorithme garantit la validité technique |
| **Pertinence** | LLM filtre par le sens sémantique |
| **Réduction du bruit** | De 331 FDs à 16 candidates à 12 significatives |
| **Classification** | Distingue key, business_rule, derived |
| **Explicabilité** | Le LLM fournit des raisons |

## 6.8 Conclusion Task 4

> **Le pipeline hybride combine efficacement la rigueur algorithmique et l'intelligence sémantique. Sur 331 FDs découvertes algorithmiquement, le pipeline identifie 12 FDs significatives (3.6%) avec des explications claires. C'est une réduction de 96% du volume à analyser tout en conservant les FDs importantes.**

---

# 7. Conclusion Générale

## 7.1 Synthèse des résultats

| Task | Objectif | Résultat clé |
|------|----------|--------------|
| **Task 1** | Découverte algorithmique | 331 FDs trouvées (BRIDGES: 325, IRIS: 6) |
| **Task 2** | Analyse sémantique LLM | Catégorisation: key, business_rule, derived, accidental |
| **Task 3** | Étude échantillonnage | **9% de faux positifs** sur échantillons |
| **Task 4** | Pipeline hybride | **12 FDs significatives** sur 16 candidates (75%) |

## 7.2 Apprentissages principaux

### 7.2.1 Sur les algorithmes de découverte

- Les algorithmes trouvent **toutes** les FDs valides techniquement
- Ils ne distinguent pas trivial/significatif
- Le volume de FDs peut être écrasant (325 pour un dataset de 108 lignes)

### 7.2.2 Sur les LLMs

- Capables d'évaluer la **signification sémantique**
- Distinguent correctement clés primaires vs règles métier
- Limitées : peuvent être trompés par les corrélations accidentelles

### 7.2.3 Sur l'échantillonnage

- **Dangereux** : 9% de faux positifs
- L'échantillon stratifié est meilleur que l'aléatoire
- **Toujours valider sur le dataset complet**

### 7.2.4 Sur l'approche hybride

- Combine le meilleur des deux mondes
- Réduit le volume de 96% (331 → 12)
- Fournit des explications

## 7.3 Recommandations pratiques

### Pour les praticiens de la qualité des données :

1. **Ne jamais faire confiance aux FDs sur échantillon** - Toujours valider sur le complet

2. **Utiliser une approche hybride** - Algorithme pour découvrir, LLM pour filtrer

3. **Classifier les FDs** - key vs business_rule vs derived vs accidental

4. **Documenter les FDs significatives** - Avec explications du LLM

5. **Considérer les FDs approximatives** - Une FD à 95% peut être plus utile qu'une FD triviale à 100%

## 7.4 Limites de l'étude

| Limite | Impact | Mitigation possible |
|--------|--------|---------------------|
| Nombre de datasets | Généralisation limitée | Tester sur plus de datasets |
| Taille LHS limitée à 2 | FDs complexes non découvertes | Augmenter max_lhs_size |
| Un seul LLM testé | Biais possible | Comparer plusieurs LLMs |
| Coût API | Scalabilité limitée | Modèles locaux |

## 7.5 Perspectives futures

1. **Multi-LLM** : Comparer Claude, GPT-4, Gemini pour la robustesse
2. **FDs conditionnelles** : Étendre aux CFDs
3. **Feedback loop** : Améliorer le LLM avec les validations
4. **Interface utilisateur** : Outil interactif pour l'exploration

## 7.6 Message final

> **La découverte de dépendances fonctionnelles significatives nécessite une approche hybride combinant la précision des algorithmes classiques et l'intelligence sémantique des LLMs. Ni l'un ni l'autre seul ne suffit : les algorithmes trouvent tout mais sans discernement, les LLMs comprennent le sens mais peuvent halluciner. Ensemble, ils permettent d'identifier les vraies contraintes métier dans les données.**

---

# 8. Annexes

## A. Technologies utilisées

| Technologie | Version | Usage |
|-------------|---------|-------|
| Python | 3.9+ | Langage principal |
| Pandas | 2.0+ | Manipulation de données |
| NumPy | <2.0 | Calculs numériques |
| Anthropic API | Claude 3 Haiku | Analyse sémantique |
| Matplotlib | 3.7+ | Visualisations |
| python-dotenv | 1.0+ | Gestion des clés API |
| Jupyter | 7.0+ | Notebooks interactifs |

## B. Structure du projet

```text
Projet_Data_Quality/
│
├── Datasets/
│   ├── iris/
│   │   └── iris.data
│   ├── pittsburgh+bridges/
│   │   └── bridges.data.version1
│   ├── abalone/
│   │   └── abalone.data
│   ├── breast+cancer+wisconsin+original/
│   │   └── breast-cancer-wisconsin.data
│   └── nursery/
│       └── nursery.data
│
├── notebooks/
│   ├── Task1_FD_Analysis.ipynb
│   ├── Task2_LLM_Semantic_Analysis.ipynb
│   ├── Task3_Sampling_FD_Hypotheses.ipynb
│   └── Task4_Hybrid_Pipeline.ipynb
│
├── results/
│   ├── task1_all_fds.json
│   ├── task1_fd_statistics.png
│   ├── task1_summary.csv
│   ├── task2_comparison.csv
│   ├── task2_gemini_responses.json
│   ├── task3_validation_results.csv
│   ├── task3_llm_responses.json
│   ├── task3_parsed_fds.json
│   ├── task4_hybrid_results.csv
│   ├── task4_hybrid_results.json
│   └── task4_hybrid_analysis.png
│
├── .env                    # Clés API (non versionné)
├── .gitignore
└── README.md
```

## C. Configuration de l'environnement

```bash
# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install pandas numpy matplotlib anthropic python-dotenv jupyter

# Configurer les clés API
echo "CLAUDE_API_KEY=sk-ant-..." > .env
```

## D. Références

1. **UCI Machine Learning Repository** - Datasets utilisés
   - https://archive.ics.uci.edu/

2. **Anthropic Claude API** - Documentation
   - https://docs.anthropic.com/

3. **Algorithmes de découverte de FDs**
   - TANE: Huhtala et al. (1999)
   - FUN: Novelli & Cicchetti (2001)

4. **Cours de Qualité des Données**
   - Master 2 IASD, Université Paris-Dauphine

---

*Rapport généré dans le cadre du projet de Qualité des Données*

*Master 2 IASD - Université Paris-Dauphine*

*Février 2026*
