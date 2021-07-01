import re
from interests import interest_dict

def valid_email(email):
    email_regex = re.compile(r"[^@]+@[^@]+\.[^@]+")
    return email_regex.match(email)

def valid_date(date):
    x = re.findall('\d{2}/\d{2}/\d{4}', date)
    if len(x)!=1:
        return False
    else:
        if x[0]==date:
            return True
        else:
            return False

def validate_interests(interest_list):
    #Check type
    if not isinstance(interest_list, list):
        return False
    #Check length
    amount_interest = len(interest_list)
    if amount_interest < 3 or amount_interest > 5:
        return False
    #Check that values are in the category
    for elem in interest_list:
        if elem not in interest_dict:
            return False
    return True
