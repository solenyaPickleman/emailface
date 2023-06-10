import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import transformers

model_name = "google/flan-t5-large"

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = T5ForConditionalGeneration.from_pretrained(model_name, quantization_config=quant_config, device_map={"":0})
model.gradient_checkpointing_enable()


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules= ["q", "v"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, config)



#setup dataset 
df = pd.read_parquet('~/dataset.parquet')
df['is_junk'] = df.apply(lambda x: all([x[k] == False for k in ['is_travel','is_ad', 'is_virtual']])  and all([x[k]=='N/A' for k in ['who','where','who_from','when','summary']]), axis=1)
df = df[~df['is_junk']]

#build dataset : 
questions = [
     'Text : %s. \n\n Based on the previous text : Yes or No - Does this document contain confirmation or planning for a reservation, booking, flight, hotel, or trip? Answer : ',
     'Text : %s. \n\n Based on the previous text : What date and time is the reservatioan? Answer : ',
     'Text : %s. \n\n Based on the previous text : What location is the reservation for? Answer  : ',
     'Text : %s. \n\n Based on the previous text : Who is the reservation for? Answer : ',
     'Text : %s. \n\n Based on the previous text : Who is the message from? Answer : ',
     'Text : %s. \n\n Based on the previous text : Summarize the text in one sentence in English: ',
     'Text : %s. \n\n Based on the previous text : Yes or no - is this a mass promotional message, or concerning a discount or offer from a store? Answer: ',
     'Text : %s. \n\n Based on the previous text : Yes or no - is this about a virtual event? Answer  : '
]
q_index = ['is_travel','when','where','who','who_from','summary','is_ad','is_virtual']
prompt_dataset = []
for row in df.to_dict('records') : 
    for i,q in enumerate(questions) : 
        answer = row[q_index[i]] 
        if q_index[i].startswith('is_') : 
            answer = 'Yes' if answer else 'No' 
        rec = {
            'text' : q % row['content'][:1500] ,
            'answer' : answer 
        }
        prompt_dataset.append(rec) 

df = pd.DataFrame(prompt_dataset)
df = df[df['answer']!='N/A']
df = df[df['answer']!='N/A.']

df['data'] = df['text'] + df['answer']
df = df[['data']]

train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42  )
train_ = Dataset.from_pandas(train)
test_ = Dataset.from_pandas(test)

train_ = train_.map(lambda samples: tokenizer(samples["data"]), batched=True)
test_ = test_.map(lambda samples: tokenizer(samples["data"]), batched=True)



tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=2,
        max_steps=20,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
