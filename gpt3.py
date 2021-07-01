from config_parser import get_config_dict
import argparse
import os
import openai

class GPT3Handler():
    def __init__(self,
        engine="davinci",
        temperature=0.50,
        max_tokens=100,
        frequency_penalty=0.15,
        debug=False,
        default_prompt="I am a highly intelligent and funny question answering bot. If you ask me a question that is rooted in truth, I will give you the answer." \
        " If you ask me a question that is nonsense, trickery or has no clear answer, I will respond with a funny response.\n\nQ: What is human life expectancy" \
        " in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: What is the square root of banana?\nA: 42.\n\nQ: "):
        #Init parameters
        self.api_key = get_config_dict()["gpt3"]["api_key"]
        openai.api_key = self.api_key
        self.start_sequence = "\nA:"
        self.restart_sequence = "\n\nQ: "
        self.default_prompt = default_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.engine = engine
        self.frequency_penalty = frequency_penalty
        self.debug=debug
        #To allow to save parameters
        self.params_dict = {}

    def ask_question(self, question):
        prompt = self.default_prompt+question+"\nA: "
        if self.debug:
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
        return answer_text

if __name__ == "__main__":
    gpt3 = GPT3Handler(debug=True)
    question = input("What will you ask GPT3?:")
    print("You asked: {}".format(question))
    response = gpt3.ask_question(question)
    print("Asked GPT3: {}".format(question))
    print("Response: {}".format(response))
