# Tableau de bord Fitbit Analytics

Transformez vos données Google Fitbit Takeout en visualisations de santé détaillées et organisées avec une granularité supérieure à celle de l'application Fitbit standard.

## Fonctionnalités

- **Visualisation continue des données** - Voyez chaque point de données (fréquence cardiaque seconde par seconde, activité minute par minute)
- **Plus de détails que l'application Fitbit** - L'application Fitbit agrège les données ; ce tableau de bord affiche les mesures brutes continues
- **Graphiques interactifs** - Zoomez, déplacez et explorez vos données de santé avec Plotly
- **Export PDF** - Générez des rapports HTML imprimables avec des graphiques intégrés
- **Plusieurs sources de données :**
  - Fréquence cardiaque (continue, toutes les quelques secondes)
  - Phases et durée du sommeil
  - Taux d'oxygène dans le sang (SpO2)
  - Variabilité de la fréquence cardiaque (HRV)
  - Pas et activité
  - Scores de stress

## Démarrage rapide

1. **Exportez vos données Fitbit :**
   - Allez sur [Export de données Fitbit](https://www.fitbit.com/settings/data/export)
   - Demandez vos données (peut prendre jusqu'à 24 heures)
   - Téléchargez le fichier ZIP

2. **Lancez le tableau de bord :**
   ```bash
   pip install -r requirements.txt
   streamlit run health_dashboard.py
   ```

3. **Téléchargez vos données :**
   - Téléchargez le fichier `Takeout.zip` dans la barre latérale
   - Ou placez votre dossier `Takeout*/Fitbit` dans le même répertoire

4. **Générez un rapport PDF :**
   - Cliquez sur "Generer rapport" dans la barre latérale
   - Téléchargez le fichier HTML
   - Ouvrez-le dans le navigateur et appuyez sur Ctrl+P → Enregistrer en PDF

## Pourquoi ce projet existe

L'application Fitbit standard affiche des résumés quotidiens et un historique limité. Ce tableau de bord fournit :
- **Résolution temporelle complète** - Chaque mesure de fréquence cardiaque, pas seulement les moyennes
- **Tendances à long terme** - Visualisez toutes vos données à la fois, pas jour par jour
- **Analyse personnalisée** - Alertes santé et métriques calculées à partir de votre jeu de données complet
- **Rapports portables** - Générez des PDF pour vos archives ou votre professionnel de santé

## Prérequis

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- Kaleido (pour l'export PNG)

## Licence

Licence MIT
