import os
import sys
sys.path.append(os.getcwd())
from utils import read_txt
import random
#Cache question list for server
path_question_list = "questions/questions.txt"
question_list = read_txt(path_question_list)
