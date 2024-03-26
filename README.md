DiversifyQG
====
The Pytorch implementation of Diversifying Question Generation over Knowledge Base via External Natural Questions(LERC-COLING 2024).

Requirements
====
1.Environments
* Create a virtual environment by running the following command:
```
$ conda env create --name=DiversifyQG --file=environment.yml
```
* Activate the environment using:
```
$ conda activate DiversifyQG
```
2.Dataset

(1) WQ : `dataset/` contains files for WQ dataset.

(2) PQ : `dataset/` contains files for PQ dataset.

Specifically, `WQ/` and `PQ/` contain the following files:
* `train.json`, `dev.json` and `test.json` are the data for for train, dev and test, respectively.

* `train_question_gold.txt`, `val_question_gold.txt` and `test_question_gold.txt` are the gold questions for train data, dev data and test data, respectively. 

* `pseundo.txt` is the pseudo questions from natural questions.

How to run 
====
1.Prepare data.
```
$ CUDA_VISIBLE_DEVICES=0 python preprocess.py --input_dir dataset/WQ --output_dir './output_WQ' --model_name_or_path 'facebook/bart-base'
```
2.To train the example, execute:
```
$ CUDA_VISIBLE_DEVICES=0,1 python train_main_finetune.py --num_train_epochs 30 --input_dir dataset/WQ --output_dir './output_WQ' --learning_rate 5e-5 --batch_size 8 --model_name_or_path 'facebook/bart-base'
```
3.To infer the example, execute:
```
$ CUDA_VISIBLE_DEVICES=0,1 python infer.py --input_dir dataset/WQ --output_dir './output_WQ' --batch_size 8 --model_name_or_path 'facebook/bart-base'
```
