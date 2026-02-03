"""
Application Streamlit - Projet Qualité des Données
Découverte de Dépendances Fonctionnelles avec Algorithmes + LLM
M2 IASD - Paris Dauphine
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from itertools import combinations
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="FD Discovery - Projet Data Quality",
    page_icon="🔍",
    layout="wide"
)

# Titre principal
st.title("🔍 Découverte de Dépendances Fonctionnelles")
st.markdown("### Approche Hybride : Algorithmes + LLM")
st.markdown("**M2 IASD - Université Paris-Dauphine**")
st.markdown("---")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une section :",
    ["🏠 Accueil",
     "📊 Task 1: Découverte Algorithmique",
     "🤖 Task 2: Analyse LLM",
     "📉 Task 3: Échantillonnage",
     "🔄 Task 4: Pipeline Hybride",
     "📈 Résultats Finaux"]
)

# ============================================
# Fonctions utilitaires
# ============================================

@st.cache_data
def load_datasets():
    """Charge tous les datasets"""
    datasets = {}

    # IRIS
    iris_path = 'Datasets/iris/iris.data'
    if os.path.exists(iris_path):
        datasets['iris'] = pd.read_csv(iris_path, header=None,
                                        names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

    # BRIDGES
    bridges_path = 'Datasets/pittsburgh+bridges/bridges.data.version1'
    if os.path.exists(bridges_path):
        datasets['bridges'] = pd.read_csv(bridges_path, header=None,
                                           names=['IDENTIF', 'RIVER', 'LOCATION', 'ERECTED', 'PURPOSE',
                                                  'LENGTH', 'LANES', 'CLEAR-G', 'T-OR-D', 'MATERIAL',
                                                  'SPAN', 'REL-L', 'TYPE'])

    # ABALONE
    abalone_path = 'Datasets/abalone/abalone.data'
    if os.path.exists(abalone_path):
        datasets['abalone'] = pd.read_csv(abalone_path, header=None,
                                           names=['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                                                  'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'])

    # NURSERY
    nursery_path = 'Datasets/nursery/nursery.data'
    if os.path.exists(nursery_path):
        datasets['nursery'] = pd.read_csv(nursery_path, header=None,
                                           names=['parents', 'has_nurs', 'form', 'children',
                                                  'housing', 'finance', 'social', 'health', 'class'])

    return datasets

def check_fd(df, lhs_cols, rhs_col):
    """Vérifie si une FD tient"""
    if isinstance(lhs_cols, str):
        lhs_cols = [lhs_cols]

    grouped = df.groupby(list(lhs_cols))[rhs_col].nunique()
    violations = (grouped > 1).sum()
    total_groups = len(grouped)

    if total_groups == 0:
        return False, 0, 0

    validity_rate = ((total_groups - violations) / total_groups) * 100
    holds = violations == 0

    return holds, validity_rate, violations

def discover_fds(df, min_validity=90, max_lhs_size=2):
    """Découvre les FDs"""
    columns = list(df.columns)
    fds = []

    for lhs_size in range(1, max_lhs_size + 1):
        for lhs_cols in combinations(columns, lhs_size):
            lhs_cols = list(lhs_cols)
            for rhs_col in columns:
                if rhs_col in lhs_cols:
                    continue
                holds, validity, violations = check_fd(df, lhs_cols, rhs_col)
                if validity >= min_validity:
                    fds.append({
                        'lhs': ', '.join(lhs_cols),
                        'rhs': rhs_col,
                        'fd_string': f"{', '.join(lhs_cols)} → {rhs_col}",
                        'validity': validity,
                        'violations': violations
                    })

    return fds

# ============================================
# Pages
# ============================================

if page == "🏠 Accueil":
    st.header("Bienvenue dans le Projet Qualité des Données")

    st.markdown("""
    ## 🎯 Objectif du Projet

    Combiner les **algorithmes classiques** de découverte de dépendances fonctionnelles
    avec les **Large Language Models (LLMs)** pour identifier les FDs **significatives**.

    ## 📋 Les 4 Tâches
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **Task 1 : Découverte Algorithmique**
        - Algorithme de découverte de FDs
        - Calcul de validité
        - FDs exactes et approximatives
        """)

        st.success("""
        **Task 2 : Analyse Sémantique LLM**
        - Évaluation par Claude 3 Haiku
        - Catégorisation : key, business_rule, derived, accidental
        - Score sémantique 0-10
        """)

    with col2:
        st.warning("""
        **Task 3 : Échantillonnage**
        - Échantillons aléatoires vs stratifiés
        - Détection des faux positifs
        - **9% de faux positifs détectés !**
        """)

        st.error("""
        **Task 4 : Pipeline Hybride**
        - Combinaison Algorithme + LLM
        - Score hybride = (technique + sémantique) / 2
        - **Réduction de 96% du volume**
        """)

    st.markdown("---")
    st.markdown("### 📊 Datasets utilisés")

    datasets = load_datasets()
    for name, df in datasets.items():
        st.write(f"**{name.upper()}** : {df.shape[0]} lignes × {df.shape[1]} colonnes")


elif page == "📊 Task 1: Découverte Algorithmique":
    st.header("Task 1 : Découverte Algorithmique des FDs")

    datasets = load_datasets()

    # Sélection du dataset
    dataset_name = st.selectbox("Choisir un dataset :", list(datasets.keys()))
    df = datasets[dataset_name]

    st.subheader(f"Aperçu de {dataset_name.upper()}")
    st.dataframe(df.head(10))

    col1, col2 = st.columns(2)
    with col1:
        min_validity = st.slider("Validité minimum (%)", 80, 100, 90)
    with col2:
        max_lhs = st.slider("Taille max du LHS", 1, 3, 2)

    if st.button("🔍 Découvrir les FDs", type="primary"):
        with st.spinner("Analyse en cours..."):
            fds = discover_fds(df, min_validity=min_validity, max_lhs_size=max_lhs)

        st.success(f"✅ {len(fds)} FDs découvertes !")

        if fds:
            fds_df = pd.DataFrame(fds)
            fds_df = fds_df.sort_values('validity', ascending=False)

            st.subheader("FDs découvertes")
            st.dataframe(fds_df[['fd_string', 'validity', 'violations']], use_container_width=True)

            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                exact = len([f for f in fds if f['validity'] == 100])
                st.metric("FDs exactes (100%)", exact)
            with col2:
                approx = len([f for f in fds if f['validity'] < 100])
                st.metric("FDs approximatives", approx)
            with col3:
                st.metric("Total", len(fds))


elif page == "🤖 Task 2: Analyse LLM":
    st.header("Task 2 : Analyse Sémantique avec LLM")

    st.markdown("""
    Le LLM (Claude 3 Haiku) évalue chaque FD et attribue :
    - Un **score** de 0 à 10
    - Une **catégorie** : key, business_rule, derived, accidental, meaningless
    """)

    # Charger les résultats pré-calculés
    results_path = 'results/task4_hybrid_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)

        st.subheader("Résultats de l'analyse sémantique")

        results_df = pd.DataFrame(results)

        # Afficher par catégorie
        categories = results_df['category'].unique()

        for cat in ['key', 'business_rule', 'derived', 'accidental', 'meaningless']:
            if cat in categories:
                cat_df = results_df[results_df['category'] == cat]

                if cat == 'key':
                    st.success(f"🔑 **{cat.upper()}** ({len(cat_df)} FDs)")
                elif cat == 'business_rule':
                    st.info(f"📋 **{cat.upper()}** ({len(cat_df)} FDs)")
                elif cat == 'derived':
                    st.warning(f"🔗 **{cat.upper()}** ({len(cat_df)} FDs)")
                else:
                    st.error(f"❌ **{cat.upper()}** ({len(cat_df)} FDs)")

                for _, row in cat_df.iterrows():
                    st.write(f"  • `{row['fd']}` - Score: {row['semantic_score']}/10 - {row.get('reason', '')}")
    else:
        st.warning("Résultats non disponibles. Exécutez d'abord le notebook Task 4.")

    st.markdown("---")
    st.subheader("💡 Catégories expliquées")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Catégorie | Description |
        |-----------|-------------|
        | **key** | Clé primaire / identifiant unique |
        | **business_rule** | Règle métier significative |
        | **derived** | Attribut calculé ou dérivé |
        """)
    with col2:
        st.markdown("""
        | Catégorie | Description |
        |-----------|-------------|
        | **accidental** | Corrélation sans causalité |
        | **meaningless** | Pas de sens sémantique |
        """)


elif page == "📉 Task 3: Échantillonnage":
    st.header("Task 3 : Échantillonnage et Faux Positifs")

    st.markdown("""
    ## ⚠️ Le problème de l'échantillonnage

    Une FD peut sembler **vraie sur un échantillon** mais être **fausse sur le dataset complet**.
    """)

    # Charger les résultats
    results_path = 'results/task3_validation_results.csv'
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)

        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total FDs testées", len(results_df))
        with col2:
            tp = len(results_df[results_df['category'] == 'true_positive'])
            st.metric("✅ Vraies positives", tp)
        with col3:
            fp = len(results_df[results_df['category'] == 'false_positive'])
            st.metric("❌ Faux positifs", fp, delta=f"{fp/len(results_df)*100:.1f}%", delta_color="inverse")
        with col4:
            fn = len(results_df[results_df['category'] == 'false_negative'])
            st.metric("Fausses", fn)

        st.markdown("---")
        st.subheader("🚨 Les Faux Positifs détectés")

        false_positives = results_df[results_df['category'] == 'false_positive']

        if len(false_positives) > 0:
            for _, fp in false_positives.iterrows():
                st.error(f"""
                **FD : `{fp['fd']}`**
                - Dataset : {fp['dataset']}
                - Validité échantillon : **{fp['approx_sample']:.1f}%** ✓
                - Validité complet : **{fp['approx_full']:.1f}%** ✗
                - Violations cachées : {fp['violations_full']}
                """)

        st.markdown("---")
        st.subheader("📊 Tableau complet")
        st.dataframe(results_df[['dataset', 'sample_type', 'fd', 'holds_sample', 'holds_full', 'category']])
    else:
        st.warning("Résultats non disponibles. Exécutez d'abord le notebook Task 3.")

    st.markdown("---")
    st.info("""
    ### 💡 Leçon clé

    > **"L'échantillonnage crée des HYPOTHÈSES, pas des VÉRITÉS.
    > Toujours valider sur le dataset complet !"**
    """)


elif page == "🔄 Task 4: Pipeline Hybride":
    st.header("Task 4 : Pipeline Hybride Algorithme + LLM")

    st.markdown("""
    ## Architecture du Pipeline

    ```
    Dataset → [Algorithme] → FDs candidates → [LLM] → FDs significatives
                  ↓                              ↓
              Validité                      Score 0-10
              technique                     + Catégorie
                                                ↓
                                         Score Hybride
    ```
    """)

    st.latex(r"Score_{hybride} = \frac{Validité_{technique}/10 + Score_{sémantique}}{2}")

    # Charger les résultats
    results_path = 'results/task4_hybrid_results.csv'
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        results_df = results_df.sort_values('hybrid_score', ascending=False)

        st.markdown("---")
        st.subheader("📊 Résultats du Pipeline Hybride")

        # Classification
        col1, col2, col3 = st.columns(3)
        with col1:
            sig = len(results_df[results_df['hybrid_score'] >= 7])
            st.metric("🌟 Significatives (≥7)", sig)
        with col2:
            useful = len(results_df[(results_df['hybrid_score'] >= 5) & (results_df['hybrid_score'] < 7)])
            st.metric("👍 Utiles (5-7)", useful)
        with col3:
            ignore = len(results_df[results_df['hybrid_score'] < 5])
            st.metric("❌ À ignorer (<5)", ignore)

        st.markdown("---")

        # Tableau des résultats
        st.subheader("🏆 Classement des FDs")

        display_df = results_df[['dataset', 'fd', 'technical_validity', 'semantic_score', 'hybrid_score', 'category']].copy()
        display_df['technical_validity'] = display_df['technical_validity'].round(1)
        display_df['hybrid_score'] = display_df['hybrid_score'].round(2)

        st.dataframe(display_df, use_container_width=True)

        # Stats par catégorie
        st.markdown("---")
        st.subheader("📈 Score moyen par catégorie")

        cat_stats = results_df.groupby('category')['hybrid_score'].agg(['count', 'mean']).round(2)
        cat_stats.columns = ['Nombre', 'Score moyen']
        st.dataframe(cat_stats.sort_values('Score moyen', ascending=False))
    else:
        st.warning("Résultats non disponibles. Exécutez d'abord le notebook Task 4.")


elif page == "📈 Résultats Finaux":
    st.header("📈 Résultats Finaux et Conclusions")

    st.markdown("""
    ## 🎯 Synthèse du Projet
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Ce qu'on a appris :

        1. **Les algorithmes trouvent TOUT** mais sans discernement
        2. **Les LLMs comprennent le sens** mais peuvent halluciner
        3. **L'échantillonnage est dangereux** : 9% de faux positifs
        4. **L'approche hybride** combine le meilleur des deux
        """)

    with col2:
        st.markdown("""
        ### Chiffres clés :

        | Métrique | Valeur |
        |----------|--------|
        | FDs découvertes (algo) | 331 |
        | FDs candidates (hybride) | 16 |
        | FDs significatives | **12** |
        | Réduction du volume | **96%** |
        | Faux positifs échantillonnage | **9%** |
        """)

    st.markdown("---")

    st.success("""
    ## 💡 Message Final

    > **"La découverte de dépendances fonctionnelles significatives nécessite une approche hybride
    > combinant la précision des algorithmes et l'intelligence sémantique des LLMs."**
    """)

    st.markdown("---")
    st.markdown("### 🔗 Liens")
    st.markdown("- [GitHub Repository](https://github.com/keita223/Projet_Data_Quality)")
    st.markdown("- [Rapport complet](Rapport_Projet_Data_Quality.md)")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Projet M2 IASD**")
st.sidebar.markdown("Paris-Dauphine 2026")
