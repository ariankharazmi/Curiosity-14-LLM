import torch #pytorch for training purposes
from accelerate import Accelerator # Need this to address numpy issues
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling #OpenAI and HuggingFace packages for training, training configs, and data batch preparation
from datasets import load_dataset, concatenate_datasets # hugging face datasets

# Load the tokenizer and GPT-2 model for use, GPT-2 allows for local training/fine-tuning and no API key, so this perfect for a student project
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the padding token to the EOS token -- This keeps tokenization for sequences consistent, keep attention masking simplified
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation = True

# Load datasets from Huggingface.
comb_ds = load_dataset("yoonholee/combined-preference-dataset", split='train[:1%]', trust_remote_code=True)
pref_ds = load_dataset("OpenRLHF/preference_dataset_mixture2_and_safe_pku", split='train[:1%]', trust_remote_code=True)
com_ds = load_dataset("community-datasets/generics_kb", "generics_kb_simplewiki", split='train[:1%]', trust_remote_code=True)

# Combine dataset(s), make sure your datasets are compatible with each other
combined_dataset = concatenate_datasets([comb_ds, pref_ds, com_ds])

# Preprocess function for the combined dataset
def preprocess_function(examples): #Looks for examples as input, used as dictionary
    # Text fields can be adjusted based on data columns for dataset(s)
    text_fields = ['text', 'chosen', 'rejected', 'content', 'sentence', 'concept_name'] #Adjusted for dataset(s) columns, looks for keywords in examples dictionay
    for field in text_fields: # Goes through list of text fields (loops), if field exists it assigns value to texts and leaves loop
        if field in examples:
            texts = examples[field]
            break
    else:
        raise ValueError(f"No available text fields were found: {examples.keys()}") # If no assigned values are found, the program breaks
    # Elements MUST be Strings (or it will break)
    texts = [str(text) if text is not None else "" for text in texts]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=256) # Adjust if needed -- uniformity in sequence tokenization, longer sequences are truncated

# Print dataset (column) information (also good for debugging when your combined dataset(s) don't work together
print("Dataset columns:", combined_dataset.column_names)
print("Sample data from datasets:")
print(combined_dataset[:5])

# Tokenize the combined dataset(s)
tokenized_datasets = combined_dataset.map(preprocess_function, batched=True,
remove_columns=combined_dataset.column_names)
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask'])

# Finding (len) size of dataset(s) for future partitioning (breaking into smaller sets)
dataset_size = len(tokenized_datasets)

# Define the size of the subsets, for training sets and eval sets, good for setting sizes later
train_size = min(1000, dataset_size)
eval_size = min(200, dataset_size)

# Partition dataset into smaller sets for faster processing sepeeds and time
small_train_dataset = tokenized_datasets.shuffle(seed=42).select(range(train_size))
small_eval_dataset = tokenized_datasets.shuffle(seed=42).select(range(eval_size))

# Define training args
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Smaller batch size for faster processing speeds/time
    per_device_eval_batch_size=2,  # Smaller batch size for faster processing speeds/time
    num_train_epochs=3,  # Increase number of epochs (cycles of running through)
    weight_decay=0.01,
    save_total_limit=2,  # Number of checkpoints that will be saved
)

# Data collator function (batching samples from training set), disabling Masked Language Modelling (no BERT)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False, # No BERT
)

# Trainer is set up to work with smaller datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset, # this is all self explanatory
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train model function
trainer.train()