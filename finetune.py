import pandas as pd 
import numpy as np
import psutil
import torch
import gc 
import threading
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, T5ForConditionalGeneration

from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")



accelerator = Accelerator()
model_name_or_path = "bigscience/mt0-base"


df = pd.read_parquet('~/dataset.parquet')
df['is_junk'] = df.apply(lambda x: all([x[k] == False for k in ['is_travel','is_ad', 'is_virtual']])  and all([x[k]=='N/A' for k in ['who','where','who_from','when','summary']]), axis=1)
df = df[~df['is_junk']]

#build dataset : 
questions = [
     'Text : %s.  Yes or No - Does this document contain confirmation or planning for a reservation, booking, flight, hotel, or trip? Answer : ',
     'Text : %s. What date and time is the reservatioan? Answer : ',
     'Text : %s. What location is the reservation for? Answer  : ',
     'Text : %s. Who is the reservation for? Answer : ',
     'Text : %s. Who is the message from? Answer : ',
     'Text : %s. Summarize the text in one sentence in English: ',
     'Text : %s. Yes or no - is this a mass promotional message, or concerning a discount or offer from a store? Answer: ',
     'Text : %s. Yes or no - is this about a virtual event? Answer  : '
]
q_index = ['is_travel','when','where','who','who_from','summary','is_ad','is_virtual']
prompt_dataset = []
for row in df.to_dict('records') : 
    for i,q in enumerate(questions) : 
        answer = row[q_index[i]] 
        if q_index[i].startswith('is_') : 
            answer = 'Yes' if answer else 'No' 
        rec = {
            'text' : q % row['content'][:2000] ,
            'answer' : answer 
        }
        prompt_dataset.append(rec) 

df = pd.DataFrame(prompt_dataset)
df = df[df['answer']!='N/A']
dataset = Dataset.from_pandas(df)

# import gcld3 
# detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=2500)
# get_lang = lambda x : detector.FindLanguage(x).language
# tqdm.pandas()
# df['language'] = df['text'].progress_apply(get_lang)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

text_column = "text"
label_column = "answer"
lr = 3e-3
num_epochs = 5
batch_size = 4
seed = 42
do_test = False
set_seed(seed)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
target_max_length = 50 

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    labels = tokenizer(
        targets, max_length=target_max_length, padding="max_length", truncation='longest_first', return_tensors="pt"
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


with accelerator.main_process_first():
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=6,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )
accelerator.wait_for_everyone()

processed_datasets = processed_datasets.train_test_split (test_size=0.2, shuffle=True, seed=42 )
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["test"]
test_dataset = processed_datasets["test"]

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)

# creating model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# lr scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler
)
accelerator.print(model)

is_ds_zero_3 = False
if getattr(accelerator.state, "deepspeed_plugin", None):
    is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

for epoch in range(num_epochs):
    with TorchTracemalloc() as tracemalloc:
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
    accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
    accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
    accelerator.print(
        "GPU Total Peak Memory consumed during the train (max): {}".format(
            tracemalloc.peaked + b2mb(tracemalloc.begin)
        )
    )

    accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
    accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
    accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
    accelerator.print(
        "CPU Total Peak Memory consumed during the train (max): {}".format(
            tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
        )
    )
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

accelerator.wait_for_everyone()

model.save_pretrained('home/computron/mt0-base-5pock')
accelerator.wait_for_everyone()
