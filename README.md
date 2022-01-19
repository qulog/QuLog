This repository contains the code for the paper **"QuLog: Data-Driven Approach for Log Instruction Quality Assessment*"**. 
In the folder **code** can be found the evaluation scripts used to obtain the results. Alongside, given are the exact predictions.

The preprocessed datasets can be found in the **data** folder. 
Notably, due to proprietary issues, we do not disclose the data on the internal systems. 

To start with the code clone this GitHub repo: 

1) git clone https://github.com/qulog/QuLog.git
2) create a virtual enviorment pyton3 -m venv venv
3) Install requirements: python -m pip install -r requirments.txt
4) To check QuLog log level navigate to: ./code/level_quality/qulog_attention_nn_type1/qulog_attention_nn_type1/ (This folder contains the model
and the classes)
6) Run: python3 qulog_attention_nn_type1.py; to check the log level for an example log message. You can modify line 120 for arbitrary log instruction. 
7) To check QuLog sufficient linguistic structure navigate to: .code/ling_quality/qulog_attention_nn_type1/qulog_attention_nn_type1/ (This folder contains the model
and the classes)
8) Run: python3 qulog_attention_nn_type1.py; to check the log level for an example log message. You can modify line 327 for arbitrary log instruction.

Additionally in "./code/training_scripts/" one can find the training model scripts. Note on training: After preparing the datasets, the methods can be accessed from each script. One need to modify the paths for the correct directories where the data is located. 

Spacy note: Additionally, after installation of spacy make sure it is properly installed. If training the scripts, make sure that you have installed:
"en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0-py3-none-any.whl"


