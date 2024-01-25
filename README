# Hypernym Discovery using XLNet (SemEval 2018 Task 9)

We use the XLNet language model to solve the SemEval-2019 Task 9. We fine tune the pre-trained XLNet Large model on the data provided for the task and predict the hypernyms for the given hyponym. Scoring is done using the scoring script given by the task itself.

## Initial setup

Run the script1.sh bash file to unzip the SemEval project folder and also create some additional foldders necessary. This script will also run a preprocessing script that will use the data in the SemEval training folder and generate preprocessed text files which will be used for training.

```bash
./init.sh
```

## Training the model
The training script is given as a jupyter notebook training.ipynb. Set the parameters in the top cells of the notebook and run all the cells to fine-tune the Xlnet model. This training process will take significant time to complete (>2 hrs on Tesla T4 GPU). The notebook will also save the models at set checkpoints and also when finally done with the training.The saved model can be used to generate hypernyms for testing. 

We have saved our fine-tuned model in a drive, you can download it at https://drive.google.com/file/d/1-9_OIMIAEDjSA8IUjJm_Mo8O_NunC1Q_/view?usp=sharing . Save this model under the saved-models/ folder in the project folder and use it to generate the hypernyms.

## Generating the Hypernyms

For generating the hypernyms using the saved model from the training, run the predict.py script with appropriate arguments.
```
python3 predict.py --model="path/to/model" --input-file="path/to/data/file" --output-file="path/to/store/generated/hypernymfile" --vocab-file="path/to/corresponding/vocab/file"
```
For the tasks given in the folder, the commands to do so will be,
```
# For task 1A
python3 predict.py --model="./saved-models/model-name" --input-file="./SemEval2018-Task9/test/data/1A.english.data.txt" --output-file="./output/1A.output.txt" --vocab-file="./SemEval2018-Task9/vocabulary/1A.english.vocabulary.txt"

# For task 2A
python3 predict.py --model="./saved-models/model-name" --input-file="./SemEval2018-Task9/test/data/2A.medical.data.txt" --output-file="./output/2A.output.txt" --vocab-file="./SemEval2018-Task9/vocabulary/2A.medical.vocabulary.txt"

# For task 2B
python3 predict.py --model="./saved-models/model-name" --input-file="./SemEval2018-Task9/test/data/2B.music.data.txt" --output-file="./output/2B.output.txt" --vocab-file="./SemEval2018-Task9/vocabulary/2B.music.vocabulary.txt"

```
Generation of the hypernyms using the script takes some time, the progress for the number of hyponyms completed/ total number of hyponyms will be shown when the script runs.
## Scoring the Model
We can score the model using the generated hypernyms and the gold standard hypernyms from the SemEval task. We use the same scorer script as was provided by the task organizers.
```
python3 ./SemEval2018-Task9/task9-scorer.py path/to/gold/file path/to/generated/hypernym/file
```
For the subtasks in the SemEval task,
```
# For Task 1A
python3 ./SemEval2018-Task9/task9-scorer.py ./SemEval2018-Task9/test/gold/1A.english.gold.txt ./output/1A.output.txt > ./output/1A.score.txt

# For Task 2A
python3 ./SemEval2018-Task9/task9-scorer.py ./SemEval2018-Task9/test/gold/2A.medical.gold.txt ./output/2A.output.txt > ./output/2A.score.txt

# For Task 2B
python3 ./SemEval2018-Task9/task9-scorer.py ./SemEval2018-Task9/test/gold/2B.music.gold.txt ./output/2B.output.txt > ./output/2B.score.txt
```

The scores for each of the tasks will be saved in the files ./output/1A.score.txt, ./output/2A.score.txt, ./output/2B.score.txt respectively.
