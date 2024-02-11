import os

file_paths = [
    "./README.md",
    "./requirements.txt",
    "./config/config.yaml",
    "./config/hyper_parameters.yaml",
    "./documentation/project_documentation.docx",
    "./scripts/main.py",
    "./data/manual_test/DCIM_1141.jpg",
    "./models/model_description.txt",
    "./scripts/code/__init__.py",
    "./scripts/code/components/__init__.py",
    "./scripts/code/components/blocks.py",
    "./scripts/code/components/dataset.py",
    "./scripts/code/components/model_architecture.py",
    "./scripts/code/config/__init__.py",
    "./scripts/code/config/config.yaml",
    "./scripts/code/utilities/__init__.py",
    "./scripts/code/utilities/common_utils.py",
    "./scripts/code/logging/__init__.py",
    "./scripts/logs/__init__.py",
    "./scripts/code/pipelines/__init__.py",
    "./scripts/code/pipelines/data_gathering.py",
    "./scripts/code/pipelines/data_preparation.py",
    "./scripts/code/pipelines/model_training.py",
    "./scripts/code/pipelines/model_inferencing.py",
]

for path in file_paths:
    if not os.path.isfile(path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, "w") as file:
            file.write("")
        print(f"File created: {path}")

print("#"*100)

import os
base_path = "./"

def show_contents(base_path, depth):
    items = os.listdir(base_path)
    items = [x for x in items if "." not in x] + [x for x in items if "." in x]
    for i in range(len(items)):
        item = items[i]
        if "." in item:
            if item != items[-1]:
                print("   │"*depth + "   ├── " + item)
            else:
                print("   │"*depth + "   └── " + item)
        else:
            print("   │"*depth + "   ├── " + item + "/")
            depth+=1
            show_contents(os.path.join(base_path, item), depth)
            depth-=1
print(".\n└──")
show_contents(base_path, 0)