import tensorflow as tf
from transformers import BartTokenizer, TFBartForConditionalGeneration
from sklearn.model_selection import train_test_split





# Generate titles for new text
input_text = ""
input_encoding = tokenizer(input_text, return_tensors="tf", padding=True, truncation=True)
generated_ids = model.generate(input_encoding["input_ids"])
generated_title = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Generated Title: {generated_title}")
