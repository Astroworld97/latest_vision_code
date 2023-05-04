import json

# Define a dictionary
my_dict = {'key1': 'value1', 'key2': 'value2'}

# Open a file for writing
with open('my_dict.json', 'w') as f:
    # Write the dictionary to the file in JSON format
    json.dump(my_dict, f)
