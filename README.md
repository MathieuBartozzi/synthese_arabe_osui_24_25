# Synthèse Arabe OSUI 2024-2025

Ce projet génère un rapport structuré à partir des visites de classes de langue arabe dans le réseau OSUI. Il produit :
- une synthèse par établissement,
- une synthèse réseau,
- et un tableau de pilotage stratégique.

## Structure du projet

- `synthese_reseau_OSUI.py` : script principal d’analyse et de génération du rapport.
- `rapport_final_OSUI.docx` : rapport généré (exclu du Git).
- `raw_docs/` : dossiers Word téléchargés depuis les liens.
- `seances/` : résumés individuels par enseignant.
- `.env` : contient la clé API OpenAI (exclue du Git).

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Créer un fichier `.env` :
```
OPENAI_API_KEY=sk-...
```

## Utilisation

Lancer le script principal :
```bash
python synthese_reseau_OSUI.py
```

## Exclusions `.gitignore`

```txt
.env
rapport_final_OSUI.docx
raw_docs/
seances/
__pycache__/
```

## Auteur

[mathieu.bartozzi@mlfmonde.org](mailto:mathieu.bartozzi@mlfmonde.org)
