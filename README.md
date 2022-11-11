# Multi-task-meta-transfer-learning
## Introduction
This work aims to integrate two learning paradigms Multi-Task Learning (MTL) and meta learning, to bring together the best of both the worlds, i.e., simultaneous learning of multiple tasks, an element of MTL, and promptly adapting to new tasks, quality of meta learning. We propose Multi-task Meta Learning (MTML), an approach to enhance MTL compared to single task learning by employing meta learning. The fundamental idea of this work is to train a multi-task model, such that when an unseen task is introduced, it learns in lesser steps and gives better performance than conventional single task learning or MTL. By conducting various experiments, we demonstrate this approach on two datasets, NYU-v2, and the taskonomy dataset, on four tasks: semantic segmentation, depth  estimation, surface normal estimation and edge detection.

## Experiment configuration files
For analysing the performance of the proposed MTML algorithm, a wide range of experiments were conducted. Using this repository, 4 types of models can be trained (using the .yaml file) and tested. 
- Single task learning (exp_configs/single_task.yaml)
- Multi-task learning (exp_configs/multi_task_baseline.yaml)
- Multi-task meta learning (exp_configs/meta_multitask_baseline.yaml)
- For adding new task or finetuning on an previously trained model (exp_configs/meta_testing.yaml)

## Datasets:
Download the both datasets in the folder 'dataset' in NYU and Taskonomy dataset.
Find the NYU dataset at [here](https://drive.google.com/file/d/11pWuQXMFBNMIIB4VYMzi9RPE-nMOBU8g/view) 
Find the taskonomy dataset [here](https://github.com/StanfordVL/taskonomy/tree/master/data)

## Experiment Environment

Install the required python packages in a conda enviornment using the requirement.txt file  using --
conda create --name <env> --file requirements.txt

## Training 
To train the models use the following files- 

- For single_task training, edit the single_task.yaml file as required , activate the conda enviornment, and use the main.py file
    - python main.py --config_exp exp_configs/single_task.yaml
- For multi-task training, edit the multi_task_baseline.yaml, activate the conda enviornment,
-  and use the main.py file
    - python main.py --config_exp exp_configs/multi_task_baseline.yaml
- For training the multi-task meta learning model, edit the meta_multitask_baseline.yaml, activate the conda enviornment, and use the main_meta.py file
    - python main_meta.py --config_exp exp_configs/meta_multitask_baseline.yaml
- For fine-tuning a task or adding a new task to a previously trained model, edit the meta_testing.yaml file, and use the main_meta_test.py file 
    - python main_meta_test.py --config_exp exp_configs/meta_testing.yaml

## Testing (any model)
To test any model use the test.MTL.py file 
- python test_MTL.py --test_exp ../results/Test/path_of_the_model_for_testing



