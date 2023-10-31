from transformers import pipeline 
from pdfminer.high_level import extract_text 
import docx2txt

summarizer = pipeline("summarization", model="facebook/bart-large-xsum")

def text_extraction(file):
    if file.endswith(".pdf"):
        return extract_text(file)
    elif file.endswith(".txt"):
        with open(file, 'r', encoding='utf-8') as txt_file:
            return txt_file.read()
    else:
        resume_text = docx2txt.process(file)
    if resume_text:
        return resume_text.replace('\t','')
    return None

__path__="/home/hritik/The Israel Palestine conflict.txt"
text_extracted = text_extraction(__path__)

summary = summarizer(text_extracted, max_length = 250 , min_length = 30 , do_sample= False)

print("the summary is:", summary)
