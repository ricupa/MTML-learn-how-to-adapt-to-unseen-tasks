import numpy as np
import math
import torch






    

# def l1_inverse(depth1,depth2):
#     """
#     Computes the l1 errors between inverses of two depth maps.
#     Takes preprocessed depths (no nans, infs and non-positive values)
#     depth1:  one depth map
#     depth2:  another depth map
#     Returns: 
#         L1(log)
#     """
#     assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
#     diff = np.reciprocal(depth1) - np.reciprocal(depth2)
#     num_pixels = float(diff.size)
    
#     if num_pixels == 0:
#         return np.nan
#     else:
#         return np.sum(np.absolute(diff)) / num_pixels



# def rmse_log(depth1,depth2):
#     """
#     Computes the root min square errors between the logs of two depth maps.
#     Takes preprocessed depths (no nans, infs and non-positive values)
#     depth1:  one depth map
#     depth2:  another depth map
#     Returns: 
#         RMSE(log)
#     """
#     assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
#     log_diff = np.log(depth1) - np.log(depth2)
#     num_pixels = float(log_diff.size)
    
#     if num_pixels == 0:
#         return np.nan
#     else:
#         return np.sqrt(np.sum(np.square(log_diff)) / num_pixels)




def rmse(depth1,depth2):
    """
    Computes the root min square errors between the two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)
    depth1:  one depth map
    depth2:  another depth map
    Returns: 
        RMSE(log)
    """
    # assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 >= 0) & (depth2 >= 0)))
    
    diff = depth1 - depth2
    num_pixels = diff.size
    
    if num_pixels == 0:
        return np.nan
    else:    
        return np.sqrt(np.sum(np.square(diff)) / num_pixels)


def mean_abs_error(depth1,depth2):

    diff = np.abs(depth1 - depth2)
    num_pixels = diff.size
    if num_pixels == 0:
        return np.nan
    else:
        return np.sum(np.absolute(diff)) / num_pixels


def rel_error(pred,gt):
    rel_err = np.abs(pred - gt)/gt
    num_pixels = rel_err.size
    print('rel_err:', rel_err)

    rel_err = np.sum(rel_err)/num_pixels

    if num_pixels == 0:
        return np.nan
    else:
        return rel_err




def l1(depth1,depth2):
    """
    Computes the l1 errors between the two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)
    depth1:  one depth map
    depth2:  another depth map
    Returns: 
        L1(log)
    """
    # assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    diff = depth1 - depth2
    num_pixels = diff.size  

    return np.sum(np.absolute(diff)) / num_pixels




def get_depth_metric(pred, gt, binary_mask):

    depth_output_true = pred.masked_select(binary_mask)
    depth_gt_true = gt.masked_select(binary_mask)
    abs_err = torch.abs(depth_output_true - depth_gt_true)
    rel_err = (torch.abs(depth_output_true - depth_gt_true)+ 1e-7) / (depth_gt_true + 1e-5)
    sq_rel_err = torch.pow(depth_output_true - depth_gt_true, 2) / depth_gt_true
    abs_err = torch.sum(abs_err) / torch.nonzero(binary_mask).size(0)
    # abs_err = torch.mean(abs_err)
    rel_err = torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)
    sq_rel_err = torch.sum(sq_rel_err) / torch.nonzero(binary_mask).size(0)
    # calcuate the sigma
    term1 = depth_output_true / depth_gt_true
    term2 = depth_gt_true / depth_output_true
    ratio = torch.max(torch.stack([term1, term2], dim=0), dim=0)
    # calcualte rms
    rms = torch.mean(torch.pow(depth_output_true - depth_gt_true, 2))
    rms_log = torch.pow(torch.log10(depth_output_true + 1e-7) - torch.log10(depth_gt_true + 1e-7), 2)

    return abs_err.detach().cpu().numpy(), rel_err.detach().cpu().numpy(), sq_rel_err.detach().cpu().numpy(), ratio[0].detach().cpu().numpy(), rms.detach().cpu().numpy(), rms_log.detach().cpu().numpy()
