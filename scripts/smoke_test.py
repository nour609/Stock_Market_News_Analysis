import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('Python', sys.version)
print('pandas', pd.__version__)
print('numpy', np.__version__)

files = ["data/Combined_News_DJIA.csv", "data/RedditNews.csv", "data/upload_DJIA_table.csv"]
for f in files:
    p = Path(f)
    print(f + ':', 'FOUND' if p.exists() else 'MISSING')

if Path(files[0]).exists():
    df = pd.read_csv(files[0], nrows=2)
    print('Sample columns:', list(df.columns))

img_dir = Path('images')
img_dir.mkdir(exist_ok=True)
fig, ax = plt.subplots()
ax.plot([0,1,2],[0,1,0])
ax.set_title('smoke test')
out = img_dir / 'smoke_test.png'
fig.savefig(out)
plt.close(fig)
print('WROTE', out.exists(), out)
print('SMOKE TEST: SUCCESS')
