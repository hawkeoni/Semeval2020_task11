# Semeval 11 propaganda detection
All work done at ABBYY ARD NLP.

Team NoPropaganda place 7 on task SI, place 6 on task TI. 
[Final leaderboard](https://propaganda.qcri.org/semeval2020-task11/leaderboard.php)

*here will be a link to the paper*


*NB* it is written on [allennlp-1.0 pre-release](https://github.com/allenai/allennlp/releases/tag/v1.0-prerelease)
because it supports multi-GPU and gradient accumulation.

## Task SI
Task of binary classification of propaganda components. 
[Task description](https://propaganda.qcri.org/semeval2020-task11/data/propaganda_tasks_evaluation.pdf)


### Processing data
Data is processed to a more comfortable format. To process it run the next command:
```bash
python process_data_si.py --articles-folder datasets/train-articles/ --articles-labels-si-folder 
datasets/train-labels-task1-span-identification/ --output-filename fulltrain.txt
```
where `datasets` is the path to directory with task data. Second argument is left empty for dev & test sets.

The resulting format is as follows:

each article is separated by line

'-------------------------articleid-------------------------'

followed by lines of type:

sentence \t (without spaces) sentence offset.

Propaganda in a sentence is marked up by double square brackets `[[]]`
Example:
```
-------------------------111111111-------------------------
Next plague outbreak in Madagascar could be 'stronger': WHO     0
Geneva - The World Health Organisation chief on Wednesday said a deadly plague epidemic [[appeared]] to have been brought under control in Madagascar, but warned the next outbreak would likely be stronger. 61
```
For evaluation data was split into 5 folds. To do this run:
```
python split_into_folds_si.py fulltrain.txt 5
```
Data is split by articles not by sentences.

### Training model
The submitted model was a lasertagger over bert-base-cased encoder trained with teacher forcing and label smoothing.
Config and model weights will be available later.
Training is done via allennlp command line util.
```
allennlp train -s <serialization_dir> --include-package src <path/to/config.json>
```
Worth noting that metric used in training is token-wise f1 rather than symbol-wise f1.

### Getting predictions
Predictions are created with allennlp comandline tool.
```
allennlp predict <path/to/model.tar.gz> [test.txt|dev.txt] --output-file allennlp_pred.txt --silent 
--include-package src --batch-size 32 --use-dataset-reader --cuda-device 0

python convert_predict_to_submission_si.py allennlp_pred.txt pred.txt
```
The first command creates output in allennlp way with JSON outputs on line the next command converts it to a final
format which is used in the evaluation system.


## Task TI
Mostly all commands are the same. The only difference is the filenames ending in `ti` not on `si`.
An ensemble of models was used in the final predictions of this task. Their configs and weights will be released later.
