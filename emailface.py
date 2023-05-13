import json 
import re 
from llama_index import download_loader
from tqdm import tqdm
from time import sleep, time

import pandas as pd 
from tika import parser 
from more_itertools import chunked


smooth = re.compile('\n[\s]+\n?')

class GmailRetriveal :
    def __init__ (self) : 
        self.reader =  download_loader('GmailReader')

    def query(self, query, max_results=10) :
        loader = self.reader(query=query, max_results=max_results,  codec='utf8')
        documents = loader.load_data()

        parsed = []
        hashes = [] 
        for doc in documents :
            doc = doc.to_dict()
            try: 
                t = doc['text']
                with open('temp.eml', 'w',encoding='utf8') as file:
                    file.write(t)
                p = parser.from_file('temp.eml')['content']
                if ' ' not in p : 
                    p = doc['snippet'] if 'snippet' in doc else 'Empty'
                parsed.append(p)
                hashes.append(doc['doc_hash'])
            except Exception as e : 
                if ' ' in doc['text'] : 
                    parsed.append(doc['text'])
                else : 
                    parsed.append(doc['snippet'] if 'snippet' in doc else 'Empty')
                hashes.append(doc['doc_hash'])

        parsed = [smooth.sub('\n ', x).strip() for x in parsed]
        
        return list(zip(hashes, parsed))

# with open('/home/computron/.openai.json','r') as file: 
#     creds = json.load(file) 

search = GmailRetriveal()
docs = search.query('newer_than:1d label:inbox', max_results=500)


prompt = lambda batch : f"""
You are a chatbot looking to answer questions on provided documents. 
Answer the below questions separately. If the answer is unknown or the question is not answered in the document, response "N/A".  If you are unsure, respond "N/A"

Document : { batch }

Questions:
1. Does this document contain confirmation or planning for a reservation, booking, flight, hotel, or trip? Label product orders (like Amazon)  False. [True, False] 
2. If applicable, what date and time is the reservation? 
3. If applicable, what locations are involved? 
4. If applicable, who is the reservation for? 
5. If applicable, who is the message from?
6. Summarize the document in 10 words or less. (examples : flight to Chicago, dive trip, ad for Uber ride, etc ) 
7. Is the document an advertisement or marketing message, or does it contain information on a promotion or sale? [True, False]
8. is this a virtual event? [ True, False ]
""" 

docs = [ (a,b) for a ,b in docs if ' ' in b ]
results = []

import vertexai
from vertexai.preview.language_models import TextGenerationModel

def predict_large_language_model_sample(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = "",
    ) :
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
      model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    print(response.text)
    return response.text

run_prompt = lambda x : predict_large_language_model_sample("development-351409", "text-bison@001", 0.1, 256, 0.8, 40, prompt(x) , "us-central1")
for dochash, content in tqdm(docs) : 
    result = run_prompt(content[:5000]) 
    results.append(result) 

cols = ['is_travel','when','where','who','who_from' , 'summary', 'is_ad', 'is_virtual']
def parse(text) : 
    text = text.split('\n')
    text = [t[3:] for t in text]
    p = dict( zip ( cols , text ))
    p['is_travel'] = 'True' in p['is_travel'] or 'Yes' in p['is_travel'] 
    p['is_ad'] = 'True' in p['is_ad'] or 'Yes' in p['is_ad'] 
    p['is_virtual'] = 'True' in p['is_virtual'] or 'Yes' in p['is_virtual']
    return p 

parsed  = []
for resp in results : 
    try  : 
        parsed.append( parse(resp) )
    except Exception as e : 
        print( resp) 
        parsed.append(dict( zip ( cols , [False if x.startswith('is_') else 'N/A' for x in cols  ])))


df = pd.DataFrame(parsed)
df['hash'] = [dochash for dochash, _ in docs]
df['content'] = [c.replace('\u200c\xa0','') for _, c in docs]

#load to bigquery and print total # of records
from google.cloud import bigquery

# Construct a BigQuery client object.
table_id='development-351409.travel.email'
client = bigquery.Client(project = table_id.split('.')[0])

job_config = bigquery.LoadJobConfig(
    schema=[bigquery.SchemaField(name, bigquery.enums.SqlTypeNames.BOOLEAN) for name in ['is_travel','is_ad','is_virtual']]
)
job = client.load_table_from_dataframe(
    df, table_id, job_config=job_config
)  # Make an API request.
job.result()  # Wait for the job to complete.


# Start the query, passing in the extra configuration.
query_job = client.query(
    f"SELECT count(*) as num_rows from {table_id} limit 10 ;"
)  # Make an API request.
query_job.result()  # Wait for the job to complete.

count = pd.DataFrame([dict(x.items()) for x in query_job.result()])

print(count['num_rows'].iloc[0])
