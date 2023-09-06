# Against GPTology
This is the repositoy for the "Against GPTology" project. All code and data necessary to replicate the analysis in the paper are presented here/

## Overview
### Code

### Data Files

## Instructions
### General Setup
In a terminal execute the command `conda env update -n textgen --file textgen.yaml`. This creates a conda environment and installs the necessary packages to execute all codebooks. Then, activate the environment using `conda activate textgen`.

### Text Annotations
#### Preparation
Sample the test data from the MFRC:
  - Open jupyter lab or any other editor that can handle pythonbooks: E.g., in a terminal in your project folder, execute `jupyter lab`.
  - Open `annotations/train_bert_model/prepare_data.ipynb`. This codebook processes the MFRC data set and creates a training sample to fine-tune BERT and a separate test sample to compare BERT and ChatGPT.
  - Run all cells in the pythonbooks. This should create the following files:
        - `data/preprocessed/mfrc_eval_full.csv`
        - `data/preprocessed/mfrc_meta_sample_full.csv`
        - `data/preprocessed/mfrc_sample_full.csv`
        - `data/preprocessed/mfrc_cleaned_full.csv`
        - `data/train_test/mfrc_train_full.pkl`
      
#### BERT
1. To fine-tune BERT, open `annoations/train_bert_model/train_BERT.ipynb` and run all cells
   - You will need a GPU for this step! We used a v100 on a computing cluster but also tested it with an RTX 2070s!
   - You will have to make sure to have all GPU related packages (e.g., CUDA, CUDA Toolkit, etc) correctly installed. This should be done automatically when creating the conda environment from the .yaml file. Check this website if you encounter issues with using your GPU and tensorflow: https://www.tensorflow.org/install/pip
   - After running the file, the model will be saved in `annotations/models/mfrc_normal_full.h5`

2. Create the predictions on the test sample by running all cells in `annoations/train_bert_model/predict_BERT.ipynb`:
     - The predictions will be saved under `results/predictions/mfrc_labels_normal_full.csv`

3. We also created regular `.py` files for training and prediction to use via command line. If you are using a computing cluster with slurm, we also created exemplary `.job` files. Adjust as needed.
     - If you use the command line, the arguments for training are "mfrc", "full", "normal" (corpus, aggregate level for moral values, training type)
     - If you want to optimize the model paramters (e.g., add classification layers, change the bert model, etc), you can train using "eval" instead of "normal", which will return a cross-validated performance on the training data. In that case you also need to specify the threshold for classifying a text as a containing a moral sentiment (between 0-1).

#### ChatGPT
1. Open `annotations/codes/chatGPT_annotations.ipynb`, and add your openai API key in `openai.api_key = "" #add your openai key here`. This will allow you to use the API to request ChatGPT responses to our prompts.
     - THIS WILL CHARGE YOUR ACCOUNT!
     - Make sure that you know the prices before running (check https://openai.com/pricing)

2. Run all cells:
     - This will save the ChatGPT annotations under `results/predictions/gpt_mfrc_labels_full.csv`
  
#### Statistical Anlysis
1. Open `annotations/codes/chatGPT_performance.ipynb` and run all cells
     - This will calculate the correct/false classifications and add the annotator demographic information and save it under `../results/evals/gpt_mfrc_success_full.csv`.

2. Open `annotations/statistical_analyses/annotations_analyses.Rmd` and run all cells
     - The output of `## Evaluate' will show the logistic regression outputs for each set of annotator variables (e.g., demographics, moral values, etc). Under each regression output are the coefficients converted to percentage differences in odds. These results are presented in Table X of our work and express how each annotator characteristic is linked to the models predictions (i.e., how biased the classifier is towards said annotator characteristic).
     - The output of `## Fit Model (moral foundation ~ predictor)` will show the logistic regression of predicting each set of moral sentiment as a function of Classifier (BERT, ChatGPT, compared to humans). The results show how much more or less likely a Classifier predicts a class compared to trained human annotators (i.e., how much it over or underpredicts each moral sentiment) and is shown in Table X of our paper.
     - The output of `## Extract Coefficients` converts the coefficients above into percentage differences in odds (i.e., how much more in percent does a classifier predict a moral sentiment compared to trained humans).

### Survey Responses
1. Open `survey_predictions/code/prepare_data_gpt.ipynb` and run all cells. This will create a `data/processed/SURVEY_cleaned.csv` file for each survey in the `data/surveys` folder. In our data, some information was not collected for all participants so we filter for those participants who responded to the items of interests. *If you apply this pipeline on your own data this step will likely not be necessary or you will have to specify different items of interest in the `COLS_META` variable.*
    - The code will also generate the prompts under `data/prompts/SURVEY.pkl` for each survey. The prompts is generated from the `PROMPT_TEXT` variables and the item texts. *If you use different surveys, make sure to adjust `PROMPT_TEXT` to the respective response scales.*

2. Open `survey_predictions/code/run_prompts_gpt.ipynb` and add your openai API key to the respective variable.
    - Specify, which surveys to run in `d_list` (list the names of all surveys from `data/surveys` that you want to collect responses from). The default are the surveys we ran in our study. 
4. Run all cells. This will generate the ChatGPT responses and save them under `results/SURVEY.csv` for each SURVEY

### Statistical Analyses
1. Open `statistical_analyses/survey_analysis.Rmd` and run all cells.
    - This will calculate all group diffferences between humans and ChatGPT's survey responses, output the results as tables and save figures under `results/plots/`
    - The output of `### Demographic Group Differences` shows the differences of ChatGPT's survey responses and various demographic groups using Dunnett's Test. The test compares for each demographic variable the different levels with ChatGPT (e.g., for political orientation it compares Liberals, Moderates, Conservatives against ChatGPT). The results of this analysis are shown in Table X and Figure Y of our paper.
    - The output of `### Results` shows the regression of various human demographic variables on similarity to ChatGPT's survey responses. This expresses how much more similar ChatGPT is to a certain demographic group when responding to surveys. The results of this analysis are shown in Table X and of our paper.

2. Repeat this for any survey you are investigating (in our paper: bigfive, closure, cognition, rwa, systems_feelings; change variable `d = ` to these values).

### Prompt Sensitivity
#### Text Annotations
1. Open `annotations/codes/chatGPT_annotations_ALT.ipynb`, and add your openai API key in `openai.api_key = "" #add your openai key here`. This will allow you to use the API to request ChatGPT responses to our prompts.
     - THIS WILL CHARGE YOUR ACCOUNT!
     - Make sure that you know the prices before running (check https://openai.com/pricing)

2. Run all cells:
     - This will save the ChatGPT annotations under `results/predictions/gpt_mfrc_labels_full_ALT.csv`, one for each altered prompt (numbered; ALT1, ALT2, ...).

##### Statistical Analyses
1. Open `annotations/codes/chatGPT_performance_ALT.ipynb` and run all cells
     - This will calculate the correct/false classifications and add the annotator demographic information and save it under `../results/evals/gpt_mfrc_success_full_ALT.csv`, one for each altered prompt.

2. Open `annotations/statistical_analyses/annotations_analyses_prompting.Rmd` and run all cells
     - The output of `## Evaluate' will show the logistic regression outputs for each set of annotator variables (e.g., demographics, moral values, etc). Under each regression output are the coefficients converted to percentage differences in odds. These results are presented in Table X of our work and express how each annotator characteristic is linked to the models predictions (i.e., how biased the classifier is towards said annotator characteristic).
     - The output of `## Fit Model (moral foundation ~ predictor)` will show the logistic regression of predicting each set of moral sentiment as a function of Classifier (BERT, ChatGPT, compared to humans). The results show how much more or less likely a Classifier predicts a class compared to trained human annotators (i.e., how much it over or underpredicts each moral sentiment) and is shown in Table X of our paper.
     - The output of `## Extract Coefficients` converts the coefficients above into percentage differences in odds (i.e., how much more in percent does a classifier predict a moral sentiment compared to trained humans).
  
#### Survey Responses
1. Open `survey_predictions/code/prepare_data_gpt.ipynb` and run all cells. This will create a `data/processed/SURVEY_cleaned.csv` file for each survey in the `data/surveys` folder. In our data, some information was not collected for all participants so we filter for those participants who responded to the items of interests. *If you apply this pipeline on your own data this step will likely not be necessary or you will have to specify different items of interest in the `COLS_META` variable.*
    - The code will also generate the prompts under `data/prompts/SURVEY.pkl` for each survey. The prompts are generated from the `PROMPT_TEXT` variables and the item texts. *If you use different surveys, make sure to adjust `PROMPT_TEXT` to the respective response scales.*

2. Open `survey_predictions/code/run_prompts_gpt.ipynb` and add your openai API key to the respective variable.
    - Specify, which surveys to run in `d_list` (list the names of all surveys from `data/surveys` that you want to collect responses from). The default are the surveys we ran in our study. 
4. Run all cells. This will generate the ChatGPT responses and save them under `results/SURVEY.csv` for each SURVEY

##### Statistical Analyses
1. Open `statistical_analyses/survey_analysis_prompting.Rmd` and run all cells.
    - This will calculate all group diffferences between humans and ChatGPT's survey responses, output the results as tables and save figures under `results/plots/`
    - The output of `### Demographic Group Differences` shows the differences of ChatGPT's survey responses and various demographic groups using Dunnett's Test. The test compares for each demographic variable the different levels with ChatGPT (e.g., for political orientation it compares Liberals, Moderates, Conservatives against ChatGPT). The results of this analysis are shown in Table X and Figure Y of our paper.
    - The output of `### Results` shows the regression of various human demographic variables on similarity to ChatGPT's survey responses. This expresses how much more similar ChatGPT is to a certain demographic group when responding to surveys. The results of this analysis are shown in Table X and of our paper.

### Open-Source Pipeline
#### Preparation
- Follow https://github.com/oobabooga/text-generation-webui to install the interface for LLaMa (either use the "one-click-installer" or manually install).
- Start the interface via terminal (activate the conda environment, enter the textgen directory, run `python server.py --api`).
- In the interface, click on the "Model" tab. On the right pane, under "Download custom model or LoRA", enter "TheBloke/Luna-AI-Llama2-Uncensored-GPTQ:gptq-4bit-32g-actorder_True" and press download (this will download the model used in our studies.
- After loading is completed, on the left pane under Model choose the model on the drop down menu.
- Under "model loader" choose "ExLLama" and click on load. This will load the model so that our python script can process the prompts

##### Text Annotations
1. Open `annotations/codes/llama_annotations.ipynb`, and add your openai API key in `openai.api_key = "" #add your openai key here`. This will allow you to use the API to request ChatGPT responses to our prompts.
     - THIS WILL CHARGE YOUR ACCOUNT!
     - Make sure that you know the prices before running (check https://openai.com/pricing)

2. Run all cells:
     - This will save the ChatGPT annotations under `results/predictions/llama2_mfrc_labels_full.csv`
  
#### Statistical Anlysis
1. Open `annotations/codes/llama_performance.ipynb` and run all cells
     - This will calculate the correct/false classifications and add the annotator demographic information and save it under `../results/evals/llama2_mfrc_success_full.csv`.

2. Open `annotations/statistical_analyses/annotations_analyses_llama.Rmd` and run all cells
     - The output of `## Evaluate' will show the logistic regression outputs for each set of annotator variables (e.g., demographics, moral values, etc). Under each regression output are the coefficients converted to percentage differences in odds. These results are presented in Table X of our work and express how each annotator characteristic is linked to the models predictions (i.e., how biased the classifier is towards said annotator characteristic).
     - The output of `## Fit Model (moral foundation ~ predictor)` will show the logistic regression of predicting each set of moral sentiment as a function of Classifier (BERT, ChatGPT, compared to humans). The results show how much more or less likely a Classifier predicts a class compared to trained human annotators (i.e., how much it over or underpredicts each moral sentiment) and is shown in Table X of our paper.
     - The output of `## Extract Coefficients` converts the coefficients above into percentage differences in odds (i.e., how much more in percent does a classifier predict a moral sentiment compared to trained humans).

##### Survey Responses
1. Open `survey_predictions/code/prepare_data_llama.ipynb` and run all cells. This will create a `data/processed/SURVEY_cleaned_llama2.csv` file for each survey in the `data/surveys` folder. In our data, some information was not collected for all participants so we filter for those participants who responded to the items of interests. *If you apply this pipeline on your own data this step will likely not be necessary or you will have to specify different items of interest in the `COLS_META` variable.*
    - The code will also generate the prompts under `data/prompts/SURVEY_llama2.pkl` for each survey. The prompts are dynamically generated using `scale_meaning_dict` and the item texts. *If you use different surveys, make sure to adjust `scale_meaning_dict` to the respective response scales.*

2. Open `survey_predictions/code/run_prompts_llama2.ipynb`. Make sure that the textgen interface is running in the background.
    - Specify, which surveys to run in `d_list` (list the names of all surveys from `data/surveys` that you want to collect responses from). The default are the surveys we ran in our study. 
4. Run all cells. This will generate the LLaMa2 responses and save them under `results/SURVEY_llama2.csv` for each SURVEY.

### Statistical Analyses
1. Open `statistical_analyses/survey_analysis_llama2.Rmd` and run all cells.
    - This will calculate all group diffferences between humans and ChatGPT's survey responses, output the results as tables and save figures under `results/plots/`
    - The output of `### Demographic Group Differences` shows the differences of ChatGPT's survey responses and various demographic groups using Dunnett's Test. The test compares for each demographic variable the different levels with ChatGPT (e.g., for political orientation it compares Liberals, Moderates, Conservatives against ChatGPT). The results of this analysis are shown in Table X and Figure Y of our paper.
    - The output of `### Results` shows the regression of various human demographic variables on similarity to ChatGPT's survey responses. This expresses how much more similar ChatGPT is to a certain demographic group when responding to surveys. The results of this analysis are shown in Table X and of our paper.

2. Repeat this for any survey you are investigating (in our paper: bigfive, closure, cognition, rwa, systems_feelings; change variable `d = ` to these values).

### Comparison with top-down methods
1. Open `ccr/code/chatGPT_predictions.ipynb` and add your openai API key to the respective variable.
2. Run all cells:
     - This will save the ChatGPT predictions under `results/predictions/gpt_topdown.csv`
   
#### Statistical Analyses     
1. Open `ccr/statistical_analyses/topdown_analysis.Rmd` and run all cells.
     - This will output the statistical analysis and relevant plots and save them under `results/plots/`.
     - The output of `### Dunnett's Test (gpt & gpt_ccr vs CCR)` tests whether ChatGPT's predictions (on the item-level or construct-level) differ significantly from CCR (our topdown method).
     - The output of `### Correlation of model performances` shows the correlation between the CCR performance and ChatGPT (on the item and construct level)
     - These analyses can be repeated with different topdown methods or ChatGPT prompting styles
           - Simply run the alternative topdown method and save the resulting performance in a file analogue to the current `behavior_surve.csv` or `values_survey.csv` files. For different GPT approaches, change the prompts in the `chatGPT_predictions.ipynb` according to your respective considerations.
