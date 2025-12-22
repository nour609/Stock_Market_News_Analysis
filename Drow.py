import pandas as pd
from datetime import datetime, timedelta
import xlsxwriter

# Define the tasks with start and end dates
tasks = [
    {"Tâche": 1, "Description": "Choix du projet et validation", "Start": "2025-10-03", "End": "2025-10-10"},
    {"Tâche": 2, "Description": "Chargement et lecture des données", "Start": "2025-10-07", "End": "2025-10-14"},
    {"Tâche": 3, "Description": "Exploration des données (EDA)", "Start": "2025-10-10", "End": "2025-10-17"},
    {"Tâche": 4, "Description": "Nettoyage du texte", "Start": "2025-10-17", "End": "2025-10-24"},
    {"Tâche": 5, "Description": "Extraction de caractéristiques NLP", "Start": "2025-10-24", "End": "2025-10-31"},
    {"Tâche": 6, "Description": "Fusion des données texte + marché", "Start": "2025-10-31", "End": "2025-11-07"},
    {"Tâche": 7, "Description": "Création des jeux d'entraînement et de test", "Start": "2025-11-07", "End": "2025-11-14"},
    {"Tâche": 8, "Description": "Modélisation initiale", "Start": "2025-11-14", "End": "2025-11-21"},
    {"Tâche": 9, "Description": "Modèles avancés", "Start": "2025-11-21", "End": "2025-11-28"},
    {"Tâche": 10, "Description": "Évaluation des modèles (part 1)", "Start": "2025-11-28", "End": "2025-12-05"},
    {"Tâche": 11, "Description": "Visualisations finales", "Start": "2025-12-05", "End": "2025-12-16"},
    {"Tâche": 12, "Description": "Dépôt du code sur GitHub", "Start": "2025-12-16", "End": "2025-12-19"},
    {"Tâche": 13, "Description": "Rédaction du rapport final", "Start": "2025-12-19", "End": "2025-12-23"},
    {"Tâche": 14, "Description": "Révision finale et soumission", "Start": "2025-12-23", "End": "2025-12-29"},
]

# Parse dates
for task in tasks:
    task["Start"] = datetime.strptime(task["Start"], "%Y-%m-%d")
    task["End"] = datetime.strptime(task["End"], "%Y-%m-%d")

# Calculate overall project date range
project_start = min(task["Start"] for task in tasks)
project_end = max(task["End"] for task in tasks)

# Create date range for columns
date_range = pd.date_range(project_start, project_end)

# Start Excel
workbook = xlsxwriter.Workbook('Updated_Gantt_Chart.xlsx')
worksheet = workbook.add_worksheet('Gantt Chart')

# Formats
bold = workbook.add_format({'bold': True})
date_format = workbook.add_format({'num_format': 'dd mmm', 'align': 'center'})
bar_format = workbook.add_format({'bg_color': '#4CAF50'})

# Headers
worksheet.write('A1', 'Tâche', bold)
worksheet.write('B1', 'Description', bold)

# Write date headers
for col, date in enumerate(date_range):
    worksheet.write(0, col + 2, date, date_format)

# Write tasks and bars
for row, task in enumerate(tasks, start=1):
    worksheet.write(row, 0, task["Tâche"])
    worksheet.write(row, 1, task["Description"])

    for col, date in enumerate(date_range):
        if task["Start"] <= date <= task["End"]:
            worksheet.write(row, col + 2, '', bar_format)

# Set column widths
worksheet.set_column('A:A', 6)
worksheet.set_column('B:B', 60)
worksheet.set_column(2, 2 + len(date_range), 3)

workbook.close()
