## Meta-review
We tried our best to insert as much as possible information into the main manuscript. For the comments that were lacking space, we inserted them as supplementary material. Following the recommendations from the meta-reviewers, we upload a file: Meta_Review_Comments_Addressed_SupplementaryMaterial_ICPC_QuLog.pdf. It addresses part of the comments raised by the meta reviewers. 


# QuLog:
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


    |code 
    |--- level_quality
    |------ qulog_attention_nn_type1
    |------ qulog_svc
    |------ level_qulog_sm_rf
    |--- ling_quality
    |------ qulog_attention_nn_type1
    |------ qulog_rf
    |------ qulog_sm_svc
    |--- training_scirpts
    |------ level
    |--------- qulog_svc
    |--------- qulog_sm_rf
    |--------- qulog_attention
    |------ ling
    |--------- qulog_svc
    |--------- qulog_sm_rf
    |--------- qulog_attention
    |data 
    |--- github_repos_data
    |--- linguistic_quality_inter.csv
    |--- nine_systems_data.csv
    |--- stars_repos.csv
    |requirments.txt
    |README.md
