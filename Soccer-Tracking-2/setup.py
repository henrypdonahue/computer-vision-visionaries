import os
import torch
HOME = os.getcwd()
print(HOME)

# Enter kaggle info
os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""

# Path to kaggle json:
kaggle_path = '/home/gabe' + '/.kaggle/kaggle.json'

api_token = {"username": os.environ['KAGGLE_USERNAME'], "key": os.environ['KAGGLE_KEY']}

import json

with open(kaggle_path, 'w') as file:
    json.dump(api_token, file)

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")




