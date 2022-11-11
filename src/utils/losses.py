import torch
import torch.nn as nn
import numpy as np


# class SingleTaskLoss(nn.Module):
#     def __init__(self, loss_ft, task):
#         super(SingleTaskLoss, self).__init__()
#         self.loss_ft = loss_ft
#         self.task = task

    
#     def forward(self, pred, gt):
#         out = {self.task: torch.mean(self.loss_ft(pred[self.task], gt[self.task].cuda()))}
#         out['total'] = out[self.task]
#         return out, out  # the second return --out is of no use



# class MultiTaskLoss(nn.Module):
#     def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
#         super(MultiTaskLoss, self).__init__()
#         # assert(set(tasks) == set(loss_ft.keys()))
#         # assert(set(tasks) == set(loss_weights.keys()))
#         self.tasks = tasks
#         self.loss_ft = loss_ft
#         self.loss_weights = loss_weights

    
#     def forward(self, pred, gt):
#         out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
#         # m = min(list(out.values()))
#         # s = max(list(out.values())) - min(list(out.values()))
#         # out = {key : (val-m)/s for key,val in out.items()}
#         out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
#         return out



# class ModelLossWrapper(nn.Module):
#     def __init__(self, p, loss_ft: nn.ModuleDict, model):
#         super(ModelLossWrapper, self).__init__()
#         self.task_num = len(p['task_list'])
#         self.set_up = p['setup']
#         self.log_vars = nn.Parameter(torch.zeros((self.task_num)))
#         # self.log_vars = nn.Parameter(-3*torch.ones((self.task_num)))
#         self.loss_ft = loss_ft
#         self.tasks = p['task_list']
#         self.model = model
#         self.task = p['task_list'][0] #for single task only 


#     def forward(self, images, targets): 

#         targets = {task: val.cuda() for task, val in targets.items()}         

#         pred  = self.model(images)       

#         if self.set_up == 'multi_task':
#             out_loss = {task: self.loss_ft[task](pred[task], targets[task]) for task in self.tasks}
#             # out = {task: sum(val)/val.numel() for task,val in out_loss.items()}
#             out = out_loss
#             # log_vars = torch.clamp(self.log_vars, min =0)  # to avoid loss being negative          

#             for i, task in enumerate(self.tasks):
#                 pre = torch.exp(-self.log_vars[i])
#                 if i ==0:
#                     loss= torch.sum(pre * out_loss[task] + self.log_vars[i], -1)
                    
#                 else:
#                     loss+= torch.sum(pre * out_loss[task] + self.log_vars[i], -1)
                    
            
#             loss = torch.mean(loss)        
#             out['total'] = loss

#             print('printing log_var values: ', self.log_vars)
#             print('Losses: ', out)

#             return pred, out, self.log_vars.data.tolist()
#         else:
#             out = {self.task: torch.mean(self.loss_ft(pred[self.task], targets[self.task]))}
#             out['total'] = out[self.task]       
        
#         return pred, out, out  # the second return --out is of no use
            




