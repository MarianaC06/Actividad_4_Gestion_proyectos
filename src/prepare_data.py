from datasets import load_dataset
import pandas as pd

def sample_imdb(n=300,seed=42):
    ds=load_dataset('stanfordnlp/imdb',split='test')
    # balanced reproducible sample, labels 0=negative 1=positive
    n0=n//2; n1=n-n0
    neg=ds.filter(lambda x:x['label']==0).shuffle(seed=seed).select(range(n0))
    pos=ds.filter(lambda x:x['label']==1).shuffle(seed=seed).select(range(n1))
    frame=pd.concat([neg.to_pandas(),pos.to_pandas()],ignore_index=True)
    return frame.sample(frac=1,random_state=seed).reset_index(drop=True)

if __name__=='__main__':
    from pathlib import Path
    out=Path('data');out.mkdir(exist_ok=True)
    sample_imdb().to_csv(out/'imdb_sample.csv',index=False)
