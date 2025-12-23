# Stock Market News Analysis

Analyzes the relationship between news headlines and stock market movements using the Kaggle dataset "Daily News for Stock Market Prediction":
https://www.kaggle.com/datasets/aaron7sun/stocknews/data

## Project layout

```
├── src/                # Core modules (data loading, analysis, models)
├── data/               # Place datasets here (CSV files)
├── scripts/            # Utility scripts (report generation, inspection)
├── app.py              # Streamlit app (UI)
├── main.py             # CLI entry point for analysis
└── requirements.txt    # Pinned Python dependencies
```
## Quickstart (Windows PowerShell)
A. Create and activate a virtual environment:
python -m venv .venv

B. Activate the virtual environment:
.\.venv\Scripts\Activate.ps1

C. Upgrade pip:
python -m pip install --upgrade pip

## Quickstart (macOS / Linux)

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Dataset included and keeping data in the repo

The repository already contains a snapshot of the dataset (files in `data/`) used when this project was created:

- `data/Combined_News_DJIA.csv`
- `data/RedditNews.csv`
- `data/upload_DJIA_table.csv`

If you want the most up-to-date copy instead of the included snapshot, download the latest dataset from Kaggle and replace the files in `data/`.

Recommended (Kaggle CLI):

```bash
# install kaggle CLI and configure (see https://github.com/Kaggle/kaggle-api)
kaggle datasets download -d aaron7sun/stocknews -p data --unzip
# this will place the CSVs in `data/` — confirm filenames match those listed above
```

Manual option:

1. Visit https://www.kaggle.com/datasets/aaron7sun/stocknews/data
2. Download the CSV file(s) and move them into the `data/` folder.

Notes:
- The `data/` folder is tracked in Git and contains the snapshot files listed above. Replacing these files with newer ones will update the dataset used by the project.
- If you prefer to keep a small placeholder instead of committing CSVs, create `data/.gitkeep` and add a short `data/README.md` describing how to obtain the data.

4. Run the analysis (example):

```bash
python main.py
```

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Run the Streamlit app

After activating the virtualenv and installing requirements:

```bash
streamlit run app.py
```

## Useful scripts

- `scripts/generate_report.py`: Build static HTML report (used in `report_html/`).
- `scripts/inspect_run.py`: Lightweight script to run or inspect results programmatically.

## Notes & troubleshooting

- The `requirements.txt` file in the repo root contains pinned versions. If installation fails on macOS due to binary wheels (NumPy/Scipy), try installing a compatible version or use `conda`/`mamba`:

```bash
# example using conda
conda create -n teluqlab python=3.10
conda activate teluqlab
conda install pip
pip install -r requirements.txt
```

- To run a single script directly (without virtualenv), ensure your system Python has the packages installed.

## Files ignored by Git

The repository's `.gitignore` intentionally excludes local and large files that should not be committed:

- virtual environments: `.venv/`, `.venv_test/`, `env/`, etc.
- local environment files: `.env`, `.env.*`
- raw datasets and large artifacts in `data/` (CSV/ZIP/XLSX)
- caches and build artifacts: `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`

If you clone this repository, follow these steps to make the application runnable locally:

1. Create and activate a virtual environment (see Quickstart above).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place the CSV(s) in the `data/` directory. Because `data/` contents are ignored by Git, the folder will exist but not contain the dataset until you add it manually. You can keep an empty placeholder in the repo with:

```bash
mkdir -p data
touch data/.gitkeep
```

4. (Optional) If the project needs any secrets or credentials, create a `.env` file in the project root and add it to your local environment. Do NOT commit `.env`.

5. Run the application:

```bash
python main.py
# or for the Streamlit UI
streamlit run app.py
```

## Contributing

If you'd like changes to dependencies, tests, or CI, open an issue or send a PR. Keep dependency updates minimal and test the Streamlit UI and `main.py` behavior.

## License

See project root for licensing information.