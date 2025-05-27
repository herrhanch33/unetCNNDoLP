import argparse
import logging
import os
os.environ['WANDB_MODE'] = 'offline' # Keeps wandb offline
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb 
from evaluate import evaluate
from unet import UNet
from utils.data_loading import MoistureDataset # Your custom dataset loader
from utils.dice_score import dice_loss

dir_checkpoint = Path('./checkpoints/')

def train_model(
        model,
        device,
        args, 
        epochs: int, # No default, will come from args
        batch_size: int, # No default, will come from args
        learning_rate: float, # No default, will come from args
        save_checkpoint: bool = True,
        img_scale: float = 0.5, # No default, will come from args
        amp: bool = False, # No default, will come from args
        weight_decay: float = 1e-8, 
        momentum: float = 0.999,    
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        # ### MODIFIED ###: rgb_pattern might be ignored by MoistureDataset if it processes all PNGs
        train_dataset = MoistureDataset(data_dir=args.train_data_dir, scale=img_scale) # Removed rgb_pattern if not used by updated MoistureDataset
        val_dataset = MoistureDataset(data_dir=args.val_data_dir, scale=img_scale)     # Removed rgb_pattern
        n_train = len(train_dataset)
        n_val = len(val_dataset)
        logging.info(f'Train dataset size: {n_train}')
        logging.info(f'Validation dataset size: {n_val}')
    except Exception as e:
        logging.error(f"Error creating datasets: {e}", exc_info=True)
        sys.exit(1)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    # Ensure val_loader can handle cases where n_val might be 0 or not divisible by batch_size
    val_loader_drop_last = True if n_val > 0 and n_val % batch_size != 0 else False # Original has drop_last=True
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=val_loader_drop_last, **loader_args) if n_val > 0 else None


    # (Initialize logging) 
    if 'wandb' in sys.modules: # Check if wandb was successfully imported
        experiment = wandb.init(project='U-Net-Moisture', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp,
                 train_data_dir=args.train_data_dir, val_data_dir=args.val_data_dir) # Removed rgb_pattern if not used
        )
    else: # Simple mock for experiment.log if wandb is commented out
        class MockExperiment:
            def __init__(self): self.config = self
            def update(self, *args, **kwargs): pass
            def log(self, *args, **kwargs): pass
        experiment = MockExperiment()


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val if n_val > 0 else 'N/A (no validation data)'}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'] 
                true_masks = batch['mask'] 

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last if device.type=='cuda' else torch.contiguous_format)
                
                if model.n_classes == 1:
                    true_masks = true_masks.unsqueeze(1).to(device=device, dtype=torch.float32)
                else: 
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images) 
                    
                    if model.n_classes == 1:
                        loss = criterion(masks_pred, true_masks)
                        dice_input_pred = F.sigmoid(masks_pred).squeeze(1) 
                        dice_target_mask = true_masks.squeeze(1)          
                        loss_dice = dice_loss(dice_input_pred, dice_target_mask, multiclass=False)
                        loss += loss_dice
                    else: 
                        loss = criterion(masks_pred, true_masks)
                        loss_dice = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        loss += loss_dice

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if 'wandb' in sys.modules:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                division_step = (n_train // (5 * batch_size)) if batch_size > 0 and n_train > 0 else 0 
                if val_loader and division_step > 0 : 
                    if global_step % division_step == 0:
                        histograms = {}
                        if 'wandb' in sys.modules: # Only prepare histograms if wandb is active
                            for tag, value in model.named_parameters():
                                tag = tag.replace('/', '.')
                                if value.grad is not None: 
                                    if not (torch.isinf(value) | torch.isnan(value)).any():
                                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                    if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)
                        logging.info('Validation Dice score: {}'.format(val_score))
                        
                        if 'wandb' in sys.modules:
                            try: 
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_score,
                                    'images': wandb.Image(images[0,:3,:,:].cpu() if images.shape[1] >=3 else images[0,0,:,:].unsqueeze(0).cpu() ), 
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(torch.sigmoid(masks_pred[0]).float().cpu()), 
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except Exception as e_wandb:
                                logging.warning(f"Wandb logging failed: {e_wandb}")
        
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--train_data_dir', type=str, default='./data/train/', help='Directory for training data (e.g., ./data/train/)')
    parser.add_argument('--val_data_dir', type=str, default='./data/val/', help='Directory for validation data (e.g., ./data/val/)')
    # ### REMOVED rgb_pattern from args if MoistureDataset handles all PNGs by default ###
    # parser.add_argument('--rgb_pattern', type=str, default='_I90_', help='Pattern to identify specific RGB images (e.g., _I90_, _I0_)') 

    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs') # Changed default
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size') # Changed default
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, 
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='Downscaling factor of the images (0 to 1)') # Changed default
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision') # Default is False, add --amp to enable
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes (1 for binary segmentation)')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=4, n_classes=args.classes, bilinear=args.bilinear) 
    if device.type == 'cuda' and args.amp : # Only use channels_last with CUDA and AMP
        model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n' # Corrected typo
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if 'mask_values' in state_dict: 
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            args=args, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'If using mixed precision (--amp), try reducing batch size. '
                      'Ensure image scaling (--scale) is reducing image dimensions significantly.')
        torch.cuda.empty_cache()
        pass 
    except KeyboardInterrupt:
        logging.info('Training interrupted by user.')
        try:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'INTERRUPTED.pth'))
            logging.info('Saved interrupt checkpoint')
        except Exception as e:
            logging.error(f'Could not save INTERRUPTED.pth: {e}')
        sys.exit(0)
    except Exception as e_train:
        logging.error(f"An error occurred during training: {e_train}", exc_info=True)
        sys.exit(1)