import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
from pytorch_memlab import MemReporter

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

text = "Mount Everest is found in which mountain range?"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # torch.device("cpu")
model = model.to(device)

"""
reporter = MemReporter()
reporter.report()
"""

start_time = time.time()

preprocess_text = text.strip().replace("\n","")
tokenized_text = tokenizer.encode(preprocess_text, return_tensors="pt").to(device)
outs = model.generate(
            tokenized_text,
            max_length=10,
            num_beams=2,
            early_stopping=True)
dec = [tokenizer.decode(ids) for ids in outs]

print("--- %s seconds for total inference time ---" % (time.time() - start_time))

print("Predicted Answer: ", dec)
