from transformers import pipeline
import pandas as pd 
import gcld3
from tqdm import tqdm 

tqdm.pandas() 
detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=2500)
df = pd.read_parquet('dataset.parquet')
df['lang'] = df['content'].progress_apply(lambda x : detector.FindLanguage(x).language)
df = df[df['lang'] == 'en']

df = df.iloc[:50]

import torch 
for target in ['ru','de','ar','zh','es','pl'] : 
    torch.cuda.empty_cache()
    t = pipeline(task='translation', model='Helsinki-NLP/opus-mt-en-' + target, device=0) 
    translated = t(df['content'].values.tolist(), batch_size=32, truncation=True ) 
    translated = [x['translation_text'] for x in translated]
    df[target+'_content'] = translated

df.to_pa

