import argparse
import torch
import os
import sys
sys.path.append(os.getcwd())
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

class T5Model:
    def __init__(self,
    model_path='t5-base',
    num_return_sequences=4,
    num_beams=4,
    max_length=100,
    early_stopping=True,
    temperature=1):
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # torch.device("cpu")
        model = model.to(device)
        self.tokenizer = tokenizer
        self.model=model
        self.device=device
        self.model.eval()
        #Instantiate generation params
        self.num_return_sequences = num_return_sequences
        self.num_beams = num_beams
        self.max_length = max_length
        self.early_stopping = early_stopping
        self.temperature = temperature

    def inference(self, input):
        input_ids = self.tokenizer(input, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(input_ids=input_ids,
                                num_beams=self.num_beams,
                                num_return_sequences=self.num_return_sequences,
                                max_length=self.max_length,
                                temperature=self.temperature,
                                early_stopping=self.early_stopping)
        print(outputs)
        tokenized_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return tokenized_output

    def ask_question(self, interest_list, question):
        prompt = ""
        for interest in interest_list:
            prompt+= "i like "+interest.lower()+". "
        prompt+="<speaker2> "+question+ " <speaker1>"
        return self.inference(prompt)

def test_t5model(checkpoint_path):
    print("Loading model")
    model = T5Model(model_path=checkpoint_path)
    print("Model loaded")
    interests = "i like dogs . i like turtles . i like coffee . "
    while(True):
        #entry = input("What should I ask T5?\n")
        #prompt = interests+"<speaker2> "+entry+" <speaker1>"
        prompt = "i love to go to disney world every year . mickey mouse is my favorite character . i play with my friends on the playground . i love to sing songs from the movie frozen . i'm in the third grade . <speaker2> hello ! i was three when i learned to play guitar ! <speaker1>"
        print("Input prompt:" +prompt+"\n")
        output = model.inference(prompt)
        print("Output:"+str(output)+"\n")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Running pipeline with steps')
    parser.add_argument('--checkpoint_path', required=True, help="Path to model checkpoint")
    args = vars(parser.parse_args())
    checkpoint_path = args["checkpoint_path"]
    test_t5model(checkpoint_path)
