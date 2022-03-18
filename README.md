# Span extraction based slot-filling using attention and RNNs
In this work, we attempt to perform slot-filling using a recurrent neural network (RNN) model based on a multi-head attention mechanism. The model considers both user's utterance and slot type in determining the slot value. The attention mechanism aids in capturing key components in the dialogue related to the slot type. We compare our results to other recent state-of-the-art models based on the F1 score.

## Dataset
We use the Restaurant8K dataset to train our slot-filling model and evaluate our results. It consists of 8198 user utterances and 5 unique slot types. A total of 2975 utterances lacked any labels, and most of them accompanied multiple labels.

## Training and Evaluation
1. Step-1: Data analysis
Perform a quick check on the data.
    ```
    python utils/check_data.py 
    ```
    ```
    Dataset size: 8198
    No labels examples: 2975
    Average example text length: 7.46
    Max example text length: 36
    Slots distribution:
    people: 2164
    date: 1721
    time: 1972
    first_name: 887
    last_name: 891
    ==================================================
    Example 0
    Text: There will be 5 adults and 1 child.
    Slot: people
    Value: 5 adults and 1 child
    ==================================================

    ```
2. Step-2: Model training
Begin model training and save the best model. If you wish to use all utterances in the training set set `"include_all": true` in `config/init_config_json`. Set `"include_all": true` if you want to use only labeled utterances. Modify other parameters as per your preference.
    ```
    python train.py
    ```

3. Step-3: Evaluation 
This will use the saved model and evaluate on the test set. Model predictions are under `results/`. This scripts saves avg precision, recall, F1 scores over entire set as well as for ech example in separate json files. It also saves slot specific F1 scores.
    ```
    python evaluate.py
    ```

## Results
1. Slot specific scores
```
Slot type	    All utts	Only labelled utts
people	        74.78	    74.88
date	        86.22	    85.93
time	        98.24	    96.68
first_name	    67.36	    49.48
last_name	    35.33	    35.95
```
2. Mean scores across all utts
```
	        Train	Test
Precision	83.76	82.84
Recall	    85.17	84.25
F1	        84.3	83.32
```
3. Example of correct prediction
```
Text	    I would like a table in 2 days for me and my 4 children.				
Slot	    date				
True Value	in 2 days				
True Label	B I I				
Pred Value	in 2 days,				
Pred Label	B I I				
```
4. Example of incorrect prediction
```
Text	    Can i chnage my booking from 18 : 00 for 3 people to 17 : 45 for 4 people?			
Slot	    time				
True Value	17 : 45				
True Label	B I I				
Pred Value	00 17 : 45				
Pred Label	I B I I				
```