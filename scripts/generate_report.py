import os
import shutil
from pathlib import Path
import html

import sys
from pathlib import Path
# Ensure project root is on sys.path so we can import app when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import run_analysis


OUT_DIR = Path('report_html')
IMAGES_DIR = OUT_DIR / 'images'


def ensure_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)


def copy_image(name):
    # check both project root and images/ folder
    src = Path(name)
    alt = Path('images') / name
    if src.exists():
        shutil.copy(src, IMAGES_DIR / src.name)
        return f'images/{src.name}'
    if alt.exists():
        shutil.copy(alt, IMAGES_DIR / alt.name)
        return f'images/{alt.name}'
    return None


def write_html(filename, title, body):
    html_content = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{html.escape(title)}</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1000px }}
        header {{ margin-bottom: 20px }}
        img {{ max-width: 100%; height: auto; }}
        pre {{ background:#f6f8fa; padding:10px; border-radius:4px }}
        nav a {{ margin-right: 12px }}
      </style>
    </head>
    <body>
      <header>
        <h1>{html.escape(title)}</h1>
        <nav>
          <a href="index.html">Home</a>
          <a href="data_overview.html">Data Overview</a>
          <a href="market_analysis.html">Market Analysis</a>
          <a href="model_performance.html">Model Performance</a>
          <a href="news_analysis.html">News Analysis</a>
          <a href="statistical_insights.html">Statistical Insights</a>
        </nav>
      </header>
      {body}
    </body>
    </html>
    """
    with open(OUT_DIR / filename, 'w', encoding='utf8') as f:
        f.write(html_content)


def make_index(results):
    df = results['df']
    qr = results['quality_report']
    total = qr.get('total_rows', len(df))
    date_range = qr.get('date_range', {})
    missing = sum(qr.get('missing_values', {}).values()) if qr.get('missing_values') else df.isna().sum().sum()
    duplicates = qr.get('duplicates', 0)
    up_ratio = (df['Label'] == 1).mean() * 100

    body = f"""
    <h2>Summary</h2>
    <ul>
      <li>Total rows: {total}</li>
      <li>Date range: {html.escape(str(date_range.get('start')))} to {html.escape(str(date_range.get('end')))}</li>
      <li>Missing values: {missing}</li>
      <li>Duplicate rows: {duplicates}</li>
      <li>Up days %: {up_ratio:.1f}%</li>
    </ul>
    <p>Click the links above to view detailed pages.</p>
    """
    write_html('index.html', 'Stock Market Prediction Report', body)


def make_data_overview(results):
    df = results['df']
    sample = df.iloc[0]
    headlines = []
    for i in range(1, 26):
        h = sample.get(f'Top{i}')
        if isinstance(h, bytes):
            try:
                h = h.decode('utf8')
            except Exception:
                h = str(h)
        headlines.append(html.escape(str(h)))

    body = f"""
    <h2>Sample Headlines (first day)</h2>
    <ol>
    """
    for h in headlines:
        body += f"<li>{h}</li>\n"
    body += "</ol>"
    write_html('data_overview.html', 'Data Overview', body)


def make_market_analysis(results):
    # copy images if available
    md = []
    for img in ['topic_distribution.png', 'market_volatility.png']:
        p = copy_image(img)
        if p:
            md.append(f'<h3>{html.escape(img)}</h3><img src="{p}"/>')

    body = '<h2>Market Analysis</h2>' + '\n'.join(md)
    write_html('market_analysis.html', 'Market Analysis', body)


def make_model_performance(results):
    metrics = results.get('model_performance', {}).get('metrics', {})
    body = '<h2>Model Performance</h2>'
    body += '<h3>Metrics</h3><ul>'
    for k, v in metrics.items():
        body += f'<li>{html.escape(str(k))}: {html.escape(str(v))}</li>'
    body += '</ul>'

    for img in ['roc_curve.png', 'confusion_matrix.png', 'word_influence.png']:
        p = copy_image(img)
        if p:
            body += f'<h3>{html.escape(img)}</h3><img src="{p}"/>'

    write_html('model_performance.html', 'Model Performance', body)


def make_news_analysis(results):
    topics = results.get('topics', {})
    body = '<h2>News Analysis</h2>'
    body += '<h3>Topics</h3>'
    for t, terms in topics.items():
        body += f'<h4>{html.escape(t)} ({len(terms)} terms)</h4><ul>'
        for term, freq in terms[:20]:
            body += f'<li>{html.escape(term)} — {freq}</li>'
        body += '</ul>'

    p = copy_image('topic_distribution.png')
    if p:
        body += f'<h3>Topic distribution</h3><img src="{p}"/>'

    write_html('news_analysis.html', 'News Analysis', body)


def make_statistical_insights(results):
    sr = results.get('stats_results') or {}
    mp = results.get('model_performance') or {}
    body = '<h2>Statistical Insights</h2>'
    body += '<h3>Chi-square and CI</h3>'
    if sr:
        body += '<ul>'
        body += f"<li>Chi2: {html.escape(str(sr.get('chi2_statistic')))}</li>"
        body += f"<li>P-value: {html.escape(str(sr.get('chi2_p_value')))}</li>"
        ci = sr.get('confidence_interval')
        body += f"<li>95% CI: {html.escape(str(ci))}</li>"
        body += '</ul>'
    else:
        body += '<p>No statistical results available.</p>'

    if mp:
        body += '<h3>Cross-validation</h3>'
        body += f"<p>Mean AUC: {html.escape(str(mp.get('cv_mean')))} — Std: {html.escape(str(mp.get('cv_std')))}</p>"

    write_html('statistical_insights.html', 'Statistical Insights', body)


def main():
    print('Running analysis (this may take a while)...')
    results = run_analysis()
    print('Analysis complete — building report...')
    ensure_dirs()
    # copy some expected images
    for img in ['word_influence.png', 'topic_distribution.png', 'market_volatility.png', 'roc_curve.png', 'confusion_matrix.png']:
        copy_image(img)

    make_index(results)
    make_data_overview(results)
    make_market_analysis(results)
    make_model_performance(results)
    make_news_analysis(results)
    make_statistical_insights(results)

    print(f'Report generated at: {OUT_DIR.resolve()}/index.html')


if __name__ == '__main__':
    main()
