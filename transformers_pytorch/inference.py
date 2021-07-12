import argparse
import torch
import os
import sys
sys.path.append(os.getcwd())
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
from pytorch_memlab import MemReporter
from transformers_pytorch.tokenizer_utils import add_special_tokens_

class T5Model:
    def __init__(self, model_path='t5-base'):
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # torch.device("cpu")
        model = model.to(device)
        self.tokenizer = tokenizer
        self.model=model
        self.device=device

    def inference(self, input, num_return_sequences=1, num_beams=3, max_length=100, early_stopping=True, temperature=1):
        input_ids = self.tokenizer(input, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids=input_ids,
                                num_beams=num_beams,
                                num_return_sequences=num_return_sequences,
                                max_length=max_length,
                                early_stopping=early_stopping,
                                temperature=temperature)
        tokenized_output = self.tokenizer.batch_decode(outputs, return_tensors="pt", skip_special_tokens=True)
        return tokenized_output

    def ask_question(self, interest_list, question):
        prompt = ""
        for interest in interest_list:
            prompt+= "user likes "+interest.lower()+". "
        prompt+="<speaker2> "+question+ " <speaker1>"
        return self.inference(prompt)

def test_t5model(checkpoint_path):
    model = T5Model(model_path=checkpoint_path)
    interests = "i like computers. i like reading books. i like talking to chatbots. i love listening to classical music. "
    while(True):
        entry = input("What should I ask T5?\n")
        prompt = "<speaker2>"+entry+"<speaker1>"
        output = model.inference(prompt)
        print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running pipeline with steps')
    parser.add_argument('--checkpoint_path', required=True, help="Path to model checkpoint")
    args = vars(parser.parse_args())
    checkpoint_path = args["checkpoint_path"]
    test_t5model(checkpoint_path)
