from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(Path.cwd() / 'model' / 'en-ru-local')
model = AutoModelForSeq2SeqLM.from_pretrained(Path.cwd() / 'model' / 'en-ru-local')
def translate_phrase(phrase):
    inputs = tokenizer(phrase, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    out_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    return out_text[0]

while True:
    print(translate_phrase(input('Введите текст для перевода: ')))