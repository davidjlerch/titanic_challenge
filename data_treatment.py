import csv


def load_data(path="data/train.csv"):
    """
    load kaggle titanic data
    :param path:
    :return:
    """
    raw_data = []
    with open(path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            raw_data.append(row)
            parse_sex(row["Sex"])
    return raw_data


def treat_data(raw_data):
    """
    convert strings to floats and names to the length of the whole name (e.g. "Sage, Miss. Stella Anna" -> 3 -> normalized: 3/7)
    :param raw_data:
    :return:
    """
    treated_data = []
    labels = []
    for row in raw_data:
        treated_data.append([parse_pclass(row["Pclass"]), count_names(row["Name"]), parse_sex(row["Sex"]), parse_age(row["Age"]), parse_fare(row["Fare"]), parse_cabin(row["Cabin"]), parse_embarked(row["Embarked"])])
        labels.append(parse_survived(row["Survived"]))
    return treated_data, labels


def parse_sibsp(sibsp_string):
    return int(sibsp_string)/10


def parse_parch(parch_string):
    return int(parch_string)/10


def parse_fare(fare_string):
    return float(fare_string)/600


def parse_sex(sex_string):
    if sex_string == "female":
        return 1
    return 0


def parse_passenger_id(passenger_id_string):
    return int(passenger_id_string)/1000


def parse_pclass(pclass_string):
    return int(pclass_string)/3


def count_names(name_string):
    """
    get the amount of names (first name, second name, ... , last name) excluding "Mr.", "Mrs." and "Miss" but count academic title such as "Dr."
    :param name_string:
    :return:
    """
    if "(" not in name_string:
        count = len(name_string.split(" "))
    # workaround since names of married women are put in brackets
    elif " mrs." in name_string.lower():
        count = len(name_string.split("(")[-1].split(" "))
    else:
        count = len(name_string.split("(")[0].split(" "))
    if " mr." in name_string.lower() or " miss " in name_string.lower():
        count -= 1
    return count/7


def parse_survived(survived_string):
    return int(survived_string)


def parse_age(age_string):
    if age_string == "":
        return 0
    return float(age_string)/100


def parse_embarked(embarked_string):
    """
    convert abbreviation of city to float
    :param embarked_string:
    :return:
    """
    if embarked_string == "":
        return 0
    elif embarked_string == "C":
        return 0.3
    elif embarked_string == "Q":
        return 0.6
    elif embarked_string == "S":
        return 0.9
    else:
        raise ValueError


def parse_cabin(cabin_string):
    """
    Converting cabin number and section to float A23=1023, B104=2104, ...
    :param cabin_string: 
    :return: return normed cabin and section such as A23=0.1023, B104=0.2104, ...
    """
    cabins = ["A", "B", "C", "D", "E", "F", "G", "T"]
    if any([cabin in cabin_string for cabin in cabins]):
        if " " in cabin_string:
            cabin_string = cabin_string.split(" ")[-1]
        for letter in cabins:
            if letter in cabin_string:
                if cabin_string.split(letter)[-1] == "":
                    cabin_number_int = (cabins.index(letter) + 1) * 1000
                else:
                    cabin_number_int = (cabins.index(letter)+1)*1000 + int(cabin_string.split(letter)[-1])
                return cabin_number_int/10000
    elif cabin_string == "":
        return 0
    raise ValueError


if __name__ == "__main__":
    treated_data, labels = treat_data(load_data())
