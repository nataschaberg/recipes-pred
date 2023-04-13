# Recipe Prediction - Finetuned GPT2 distilled 🤗 model

## Context:
- project set up around the use case of: given a set of ingredients the model should provide a set of instructions to create a dish



## Structure
- `data-analysis`: contains the notebook with data cleaning, EDA, and preprocessing
- `dataset`: link to origin dataset used dor project
- `dataset-prepared`: link to files used downstream (preprocessed data, fine tuning data sets, data used in app)
- `finetuning`: 
  - contains the bash script `finetune.sh` and the huggingface script for finetuning `lm_finetuning.py`  
  - also contains the notebook used to finetune on colab `training_script.ipynb`
  - additionally a notebook which has examples of inference of base model distilGPT2 and the finetuned version
- `model-finetuned`: link to fine tuned model
- the streamlit app script is saved in root as `app.py`


## Finetuning
- model was finetuned for 2 epochs on a GPU NVIDIA A100  - took around 1 hour
- as a reference point - on the free Colab tier with Tesla T4 it would take around 5 hours (estimated based on experiments)
- all files are either in this repo or linked if you want to try the finetuning yourself



## Dataset Exploration
- can be found in the EDA section inside `data-anaylsis`
- can also be explored via Tableau dahsboard uploaded here: https://public.tableau.com/app/profile/natascha2339/viz/recipe_prediction_ver1/recipes_dashboard

![alt dashboard](https://raw.githubusercontent.com/nataschaberg/recipes-pred/main/dashboard_recipes_dataset.png)
