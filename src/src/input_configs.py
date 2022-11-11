
p={
'dataset_name': "Taskonomy",   ###Taskonomy
# 'data_dir' : "/home/ricupa/Documents/M3TL/taskonomy_dataset/",
# 'data_split': 'annotations/data_split_ids_nofolds.pkl',    
# 'class_scene_idx_name_file': 'annotations/scene_names_idx_label_63.pkl',
# 'class_object_idx_name_file': 'annotations/object_names_idx_label_100.pkl',
'setup': 'multi_task',
'backbone_model':'resnet50',   ##efficientnetb7, resnet50
'pretrained': True,
'finetuning': False,
'Experiment_name': 'trial_01_baseline_multi_task',
'task_list': ['surface_normal', 'depth_euclidean','edge_texture','segmentsemantic'],  
###'class_object', 'segmentsemantic','class_scene', 'surface_normal', 'depth_euclidean','edge_texture'
'train_batch_size' : 100,
'val_batch_size' : 32,
'test_batch_size' : 100,
'epochs' : 1,
'best_total_loss': 1,
'num_workers': 2,
'model': 'baseline',
# 'fold' : 0,     # can be from 0 to 3  for training set only  otherwise  None
# 'set': 'train',   # can be train, val and test 
'loss_weights' : {'class_scene':0.5, 'class_object': 0.5 },
'optimizer': 'adam',
'optimizer_params':{'learning_rate': 0.001, 'betas': (0.9, 0.999),'weight_decay' : 1e-4},
'scheduler': 'reduce_on_plateau',
'scheduler_params': {'lr_decay_epochs': 100, 'lr_decay_factor': 0.1, 'patience': 100 },
'threshold': 0.4,
'mode' : 'multiclass',
'comb_loss' : 'sum',
'missing_labels': True
}

