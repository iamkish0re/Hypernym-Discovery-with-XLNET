#!/bin/bash

# Unzip the SemEval Folder
unzip SemEval2018-Task9.zip

#Create Folder to save models during training.
mkdir saved-models
mkdir output

project_root=$(pwd)
semeval_folder=$project_root/SemEval2018-Task9/

#Preprocess the data
python3 preprocessing.py --input-data="$semeval_folder/training/data/combined.data.txt" --input-gold="$semeval_folder/training/gold/combined.gold.txt" --output-data="$semeval_folder/training/data/combined-preprocessed.data.txt" --output-gold="$semeval_folder/training/gold/combined-preprocessed.gold.txt"

echo "The project folder is unzipped and preprocessing of the input files is completed. You can now run the training notebook."