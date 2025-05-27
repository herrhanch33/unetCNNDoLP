import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true_original = batch['image'], batch['mask'] # Renamed mask_true to avoid confusion in scope

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format)
            # mask_true_original is [N, H, W] (long) from your MoistureDataset
            mask_true_on_device = mask_true_original.to(device=device, dtype=torch.long) 

            # predict the mask
            mask_pred_logits = net(image) # Model output [N, 1, H, W] for binary

            if net.n_classes == 1:
                assert mask_true_on_device.min() >= 0 and mask_true_on_device.max() <= 1, \
                    'True mask values should be in [0, 1] for binary segmentation'
                
                # Convert model output (logits) to binary mask (0 or 1)
                # mask_pred_logits shape: [N, 1, H, W]
                mask_pred_binary = (F.sigmoid(mask_pred_logits) > 0.5).float() # Shape: [N, 1, H, W], float, values 0.0 or 1.0

                # ### MODIFICATION HERE ###
                # Ensure mask_true has the same shape and type as mask_pred_binary for dice_coeff
                # mask_true_on_device is [N, H, W], long. Convert to [N, 1, H, W], float.
                mask_true_for_dice = mask_true_on_device.unsqueeze(1).float() 
                
                # compute the Dice score
                dice_score += dice_coeff(mask_pred_binary, mask_true_for_dice, reduce_batch_first=False)
            else: # Multi-class case
                assert mask_true_on_device.min() >= 0 and mask_true_on_device.max() < net.n_classes, \
                    'True mask indices should be in [0, n_classes-1[' # Note: original had n_classes[
                
                # Convert model output (logits) to one-hot class predictions
                # mask_pred_logits shape: [N, C, H, W]
                mask_pred_one_hot = F.one_hot(mask_pred_logits.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                
                # Convert true mask to one-hot format
                # mask_true_on_device shape: [N, H, W], long
                mask_true_one_hot = F.one_hot(mask_true_on_device, net.n_classes).permute(0, 3, 1, 2).float()
                
                # compute the Dice score, ignoring background (if applicable, often class 0)
                # The original script did [:, 1:], which assumes background is class 0 and you only score on foreground classes.
                # Adjust if your class indexing for background is different or if you want to include it.
                dice_score += multiclass_dice_coeff(mask_pred_one_hot[:, 1:], mask_true_one_hot[:, 1:], reduce_batch_first=False)

    net.train() # Set model back to training mode
    return dice_score / max(num_val_batches, 1)