# Présentation du projet — Résumé non technique

Ce projet étudie le lien entre l’actualité, les discussions en ligne et les marchés boursiers. Il collecte des articles de presse et des publications Reddit, les combine avec des données de marché, puis analyse ces informations afin d’aider à comprendre si, et comment, les informations publiques peuvent influencer les mouvements des marchés. L’objectif est de transformer des données complexes en rapports visuels clairs et accessibles à tous.

## À qui s’adresse ce projet

* Investisseurs particuliers qui souhaitent savoir si les titres de presse ou les discussions en ligne sont liés aux variations des marchés.
* Étudiants en économie ou en finance, ainsi que chercheurs, qui ont besoin d’un rapport compréhensible sur l’impact de l’actualité sur les marchés.
* Analystes et managers qui veulent des synthèses visuelles rapides sans devoir explorer des données brutes.

## Ce que fait le projet (en termes simples)

* **Collecte des informations** : rassemble des articles de presse, des publications Reddit et des données historiques de prix boursiers.
* **Recherche de signaux** : analyse les mots utilisés et le moment des publications pour voir s’ils correspondent à des mouvements du marché.
* **Mesure des performances** : utilise des modèles pour estimer dans quelle mesure les signaux liés à l’actualité expliquent ou prédisent les changements du marché.
* **Création de rapports** : génère des pages web et des graphiques faciles à lire qui résument les résultats et les tendances principales.

## Fonctionnalités clés (liste simple)

* **Collecte de données** : utilisation de données préparées provenant de l’actualité, de Reddit et de l’historique des prix boursiers.
* **Analyse de l’actualité** : identification des mots fréquents, des thèmes et du sentiment (ton positif ou négatif) dans les articles et les discussions Reddit.
* **Analyse temporelle** : comparaison entre les tendances de l’actualité et les variations des prix de marché dans le temps.
* **Évaluation des modèles** : test de modèles prédictifs simples et résumé de leurs performances.
* **Rapports automatisés** : création de rapports HTML et de graphiques lisibles directement dans un navigateur.

## Résultats produits

* Des rapports HTML interactifs dans le dossier `report_html`, présentant les analyses et les graphiques.
* Des images et graphiques montrant les tendances et comparaisons dans les dossiers `report_html/images` et `images`.
* Un tableau CSV pour le partage ou une analyse plus approfondie (pour les utilisateurs techniques) dans le dossier `data`.

## Comment consulter les résultats (pour les utilisateurs non techniques)

Ouvrez le fichier principal `report_html/index.html` dans un navigateur web. Les pages sont conçues pour permettre de lire les conclusions, observer les graphiques et comprendre les points essentiels sans aucune manipulation technique.

## Pourquoi ce projet est utile

Ce projet transforme de nombreuses sources d’information en réponses claires : quels types d’actualités apparaissent avant des mouvements de marché, si les discussions en ligne sont plutôt optimistes ou inquiètes, et dans quelle mesure ces signaux sont fiables. Cela aide à prendre des décisions plus éclairées et à mieux comprendre le lien entre l’information publique et le comportement des marchés.
