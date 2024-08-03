from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Loading GPT-2 + GPT-2 Tokenizer + Checkpoint filePATH
model_path = '/Users/kharazmimac/PycharmProjects/LLMScriptTest14/results/checkpoint-1500'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set up pipeline for text generation (relating to user prompt)
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Interactive Prompt for user, generate text based on user's entered prompt
while True:
    text = input("Enter a prompt: ")
    if text.lower() == 'exit':
        break
    result = text_generator(text, max_length=256, num_return_sequences=1)
    print(result[0]['generated_text'])
