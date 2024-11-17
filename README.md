# Instruction Tuning

## LLM_inference.ipynb
### Prerequisite
```pip install -U transformers``` <br>
```pip install peft``` <br>

This file experiments two strategies for LLM tuning: zero-shot and few-shot tuning.

## utils.py
This file contains two functions: ```get_prompt()```, which returns the prompt for the LLM, and ```get_bnb_config()```, which provides the configuration for bnb.

## QLoRa_training.py
### Prerequisite
```pip install -U transformers``` <br>
```pip install torch tqdm scikit-learn peft datasets``` <br>
### Steps to train the model
1. Data Loading <br>
Load ```train_data``` from ```train.json``` and ```public_test``` from ```public_test.json``` files.
2. Model and Tokenizer Setup 
>* Load a pre-trained tokenizer and model from the Hugging Face Model Hub (```zake7749/gemma-2-2b-it-chinese-kyara-dpo```).
>* Apply k-bit quantization (as specified by ``` get_bnb_config```) to reduce memory footprint.
3. PEFT Configuration and LoRA Setup
>* Configure the model for k-bit training using ```prepare_model_for_kbit_training```.
>* Set up the LoRA adapter.
>* Print the number of trainable parameters for tracking.
4. Data Preparation
>* Define a prompt using 20 sample instructions to help guide the model during fine-tuning.
>* Preprocess the data by tokenizing instructions and outputs and creating input masks.
>* Define a maximum input length (```max_length=2048```) for truncation.
5. Training
>* The training process uses Hugging Face's Trainer class with specified ```TrainingArguments```.
>* Evaluation every 500 steps.
6. Saving the Checkpoint <br>
After training, the script saves the adapter checkpoint in the ```adapter_checkpoint``` directory.
7. Inference <br>
For inference, the script loads the adapter checkpoint and performs inference on ```public_test``` to compute mean perplexity.

## plotCurve.ipynb
This file plots the model's learning curve during QLoRa training.

## predict.py
This file predicts ```/path/to/input.jsonl``` using the model checkpoint I trained.

## download.sh
Run this file by executing ```bash ./download.sh```. <br>
This will download the folder ```./adapter_checkpoint```, which contains the checkpoint for my model.

## run.sh
Run this file by executing ```bash ./run.sh /path/to/model /path/to/adapter_checkpoint /path/to/input.json /path/to/output.json```. <br>
This will execute the ```predict.py``` file.