from config_parser import get_config_dict
import argparse
import os
import openai
import random

class GPT3Handler():
    def __init__(self,
        engine="davinci",
        temperature=0.50,
        max_tokens=100,
        frequency_penalty=0.15,
        debug=False,
        default_prompt="I am a highly intelligent and funny question answering bot. If you ask me a question that is rooted in truth, I will give you the answer." \
        " If you ask me a question that is nonsense, trickery or has no clear answer, I will respond with a funny response.\n\nHuman: What is human life expectancy" \
        " in the United States?\nBot: Human life expectancy in the United States is 78 years.\n\nHuman: What is the square root of banana?\nBot: 42.\n\nHuman: "):
        #Init parameters
        self.api_key = get_config_dict()["gpt3"]["api_key"]
        openai.api_key = self.api_key
        self.start_sequence = "\nBot:"
        self.restart_sequence = "\n\nHuman:"
        self.default_prompt = default_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.engine = engine
        self.frequency_penalty = frequency_penalty
        self.debug=debug

    def ask_interest_question(self, question, interests):
        #Get random interest for question
        interests = [interest.lower() for interest in interests]
        rand_idx = random.randrange(len(interests))
        random_interest = interests[rand_idx]
        prompt = "Human likes "+", ".join(interests)+". Bot is a highly intelligent bot, which answers according to Human likings to please Human.\nHuman: I love "+random_interest+", do you like that?\nBot: Yes, I like "+random_interest+" quite a lot.\nHuman: "
        #Send question to GPT
        prompt = prompt+question+"\nBot: "
        if self.debug:
            print("Model running: {}".format(self.engine))
            print(prompt)
        answer = openai.Completion.create(
          engine=self.engine,
          prompt=prompt,
          temperature=self.temperature,
          max_tokens=self.max_tokens,
          top_p=1,
          frequency_penalty=self.frequency_penalty,
          presence_penalty=0,
          stop=["\n"])
        if self.debug:
            print(answer)
        answer_text = answer.choices[0].text
        return answer_text.strip()

    def ask_question(self, question):
        prompt = self.default_prompt+question+"\nBot: "
        if self.debug:
            print("Model running: {}".format(self.engine))
            print(prompt)
        answer = openai.Completion.create(
          engine=self.engine,
          prompt=prompt,
          temperature=self.temperature,
          max_tokens=self.max_tokens,
          top_p=1,
          frequency_penalty=self.frequency_penalty,
          presence_penalty=0,
          stop=["\n"])
        if self.debug:
            print(answer)
        answer_text = answer.choices[0].text
        return answer_text.strip()

if __name__ == "__main__":
    gpt3 = GPT3Handler(debug=True)
    question = input("What will you ask GPT3?:")
    print("You asked: {}".format(question))
    response = gpt3.ask_interest_question(question, ["Social Media", "Fashion", "Animals"])  #gpt3.ask_question(question)
    print("Asked GPT3: {}".format(question))
    print("Response: {}".format(response))
