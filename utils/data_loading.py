import logging
import os
from pathlib import Path
import re # For more complex parsing
import sys 

import numpy as np
import torch
from PIL import Image # Ensure Pillow is installed
from torch.utils.data import Dataset

# --- Helper function to load PNG images ---
def load_png_image(filename):
    try:
        img = Image.open(filename)
        return img
    except Exception as e:
        logging.error(f"Error opening image file {filename}: {e}")
        raise

# --- Your Custom Dataset ---
class MoistureDataset(Dataset):
    def __init__(self, data_dir: str,
                 rgb_folder_name: str = 'rgb_images',
                 dolp_folder_name: str = 'dolp_images',
                 mask_folder_name: str = 'masks',
                 scale: float = 1.0):

        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / rgb_folder_name
        self.dolp_dir = self.data_dir / dolp_folder_name
        self.mask_dir = self.data_dir / mask_folder_name
        
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids_data = [] 
        
        logging.info(f'Scanning {self.rgb_dir} for ALL .png RGB images.')
        if not self.rgb_dir.is_dir():
            raise RuntimeError(f"RGB directory not found: {self.rgb_dir}")
        if not self.dolp_dir.is_dir():
            raise RuntimeError(f"DoLP directory not found: {self.dolp_dir}")
        if not self.mask_dir.is_dir():
             raise RuntimeError(f"Mask directory not found: {self.mask_dir}")

        for rgb_filename_str in os.listdir(self.rgb_dir):
            if rgb_filename_str.endswith('.png') and not rgb_filename_str.startswith('.'):
                stem_name = Path(rgb_filename_str).stem 
                base_identifier = None
                
                match_with_i_pattern = re.match(r'(.*?)_I(?:0|90)_(\d+)$', stem_name)
                if match_with_i_pattern:
                    prefix = match_with_i_pattern.group(1) 
                    number = match_with_i_pattern.group(2) 
                    base_identifier = f"{prefix}_{number}" 
                else:
                    match_plain_pattern = re.match(r'(.*?)_(\d+)$', stem_name)
                    if match_plain_pattern:
                        prefix = match_plain_pattern.group(1) 
                        number = match_plain_pattern.group(2) 
                        base_identifier = f"{prefix}_{number}" 
                    else:
                        logging.warning(f"Could not parse base_identifier from RGB filename '{rgb_filename_str}' using defined patterns. Skipping.")
                        continue
                                
                dolp_file_path = self.dolp_dir / f"DoLP_raw_{base_identifier}.npy"
                mask_file_path = self.mask_dir / f"Mask_Otsu_{base_identifier}.png"
                rgb_full_path = self.rgb_dir / rgb_filename_str

                current_sample_data = {
                    'rgb_path': rgb_full_path,
                    'dolp_path': dolp_file_path,
                    'mask_path': mask_file_path,
                    'id': base_identifier
                }

                if dolp_file_path.exists() and mask_file_path.exists() and rgb_full_path.exists():
                    self.ids_data.append(current_sample_data)
                else:
                    logging.warning(f"Skipping RGB '{rgb_filename_str}': Missing DoLP or Mask for derived ID '{base_identifier}'. "
                                    f"Checked DoLP: {dolp_file_path} (exists: {dolp_file_path.exists()}), Mask: {mask_file_path} (exists: {mask_file_path.exists()}).")
        
        if not self.ids_data:
            raise RuntimeError(f'No suitable image triplets (RGB, DoLP, Mask) found in {self.data_dir}. Check paths and naming.')
        
        logging.info(f'MoistureDataset: Creating dataset with {len(self.ids_data)} examples from {self.data_dir}')

    def __len__(self):
        return len(self.ids_data)

    @staticmethod
    def preprocess_image(pil_img_or_np_array, scale_factor, target_size=None, is_dolp_channel=False, original_filename_for_debug=""):
        # This function now expects a PIL Image for RGB, and a NumPy array for DoLP (already loaded and float 0-1)
        try:
            if is_dolp_channel:
                # DoLP is already a NumPy array (float, 0-1 range)
                img_np_raw = pil_img_or_np_array 
                # To resize DoLP (continuous float) correctly, we'd ideally use something like scipy.ndimage.zoom
                # or convert to PIL 'F' mode, resize, then back.
                # For simplicity with PIL resize, temporarily scale to 0-255 uint8, resize, then scale back and ensure float.
                # This maintains spatial structure with NEAREST/BICUBIC based on intent.
                # Let's use BICUBIC for DoLP to preserve continuous nature after resizing.
                h_orig, w_orig = img_np_raw.shape
                if target_size:
                    newW, newH = target_size
                else:
                    newW, newH = int(scale_factor * w_orig), int(scale_factor * h_orig)
                
                assert newW > 0 and newH > 0, f'Scale {scale_factor} or target_size invalid for DoLP {original_filename_for_debug}, ({img_np_raw.shape} -> {newH}x{newW})'

                # Convert to PIL 'F' mode for float resize, or resize with OpenCV/skimage for numpy arrays
                # Easiest with PIL: scale to uint8, resize, then scale back.
                # This might lose some precision but is simpler than mode 'F' handling for some PIL versions.
                temp_pil = Image.fromarray((np.clip(img_np_raw, 0, 1) * 255).astype(np.uint8))
                resized_pil = temp_pil.resize((newW, newH), resample=Image.Resampling.BICUBIC)
                img_np = np.asarray(resized_pil).astype(np.float32) / 255.0
                img_np = np.clip(img_np, 0.0, 1.0) # Ensure range after interpolation
                return img_np[np.newaxis, ...]  # Add channel dim: [1, H, W]

            else: # For RGB (pil_img_or_np_array is a PIL Image)
                pil_img = pil_img_or_np_array
                if target_size: 
                    newW, newH = target_size
                else:
                    w, h = pil_img.size
                    newW, newH = int(scale_factor * w), int(scale_factor * h)
                
                assert newW > 0 and newH > 0, f'Scale {scale_factor} or target_size invalid for RGB image {original_filename_for_debug}, ({pil_img.size} -> {newW}x{newH})'
                
                pil_img_resized = pil_img.resize((newW, newH), resample=Image.Resampling.BICUBIC)
                img_np = np.asarray(pil_img_resized)

                if img_np.ndim == 2: img_np = img_np[np.newaxis, ...] 
                else: img_np = img_np.transpose((2, 0, 1)) 
                
                if (img_np > 1).any(): img_np = img_np / 255.0 
                return img_np.astype(np.float32)

        except Exception as e:
            logging.error(f"Error in preprocess_image for {original_filename_for_debug} (is_dolp={is_dolp_channel}): {e}", exc_info=True)
            raise

    @staticmethod
    def preprocess_mask(pil_img, scale_factor, target_size=None, original_filename_for_debug=""):
        # This function remains the same, as masks should be NEAREST and binarized
        try:
            if target_size: 
                newW, newH = target_size
            else:
                w, h = pil_img.size
                newW, newH = int(scale_factor * w), int(scale_factor * h)

            assert newW > 0 and newH > 0, f'Scale {scale_factor} or target_size invalid for mask {original_filename_for_debug}'
            
            pil_img_resized = pil_img.resize((newW, newH), resample=Image.Resampling.NEAREST)
            mask_np = np.asarray(pil_img_resized)
            
            processed_mask = (mask_np > (np.max(mask_np) / 2 if np.max(mask_np) > 0 else 0)).astype(np.int64)
            return processed_mask 
        except Exception as e:
            logging.error(f"Error in preprocess_mask for {original_filename_for_debug}: {e}", exc_info=True)
            raise

    def __getitem__(self, idx):
        sample_data_paths = self.ids_data[idx]
        target_h, target_w = -1,-1

        try:
            rgb_pil = load_png_image(sample_data_paths['rgb_path']).convert('RGB')
            
            if self.scale is not None:
                 target_w = int(self.scale * rgb_pil.width)
                 target_h = int(self.scale * rgb_pil.height)
            else: 
                 target_w, target_h = rgb_pil.width, rgb_pil.height

            # Load DoLP .npy file directly as a float array
            dolp_np_raw = np.load(sample_data_paths['dolp_path']).astype(np.float32) 
            
            # Ensure dolp_np_raw is 2D and in [0, 1] range if not already
            if dolp_np_raw.ndim == 3: 
                if dolp_np_raw.shape[0] == 1: dolp_np_raw = dolp_np_raw.squeeze(0) 
                elif dolp_np_raw.shape[2] == 1: dolp_np_raw = dolp_np_raw.squeeze(2)
                else: raise ValueError(f"DoLP .npy {sample_data_paths['dolp_path']} has 3 dims but not a single channel: {dolp_np_raw.shape}")
            assert dolp_np_raw.ndim == 2, f"DoLP .npy {sample_data_paths['dolp_path']} not 2D after potential squeeze. Shape is {dolp_np_raw.shape}"
            
            # Clip DoLP values to be strictly between 0 and 1 if they aren't already
            dolp_np_raw = np.clip(dolp_np_raw, 0.0, 1.0)

            mask_pil = load_png_image(sample_data_paths['mask_path']).convert('L') 
        except Exception as e:
            logging.error(f"Error loading files for ID {sample_data_paths.get('id', 'UNKNOWN')} (RGB: {sample_data_paths.get('rgb_path', 'N/A')}): {e}", exc_info=True)
            raise e
        
        # Preprocess (resize, convert to numpy, normalize/format)
        rgb_processed_np = self.preprocess_image(rgb_pil, self.scale, target_size=(target_w, target_h), 
                                                 is_dolp_channel=False, original_filename_for_debug=str(sample_data_paths['rgb_path']))
        # Pass the raw (but clipped 0-1) DoLP numpy array to preprocess_image
        dolp_processed_np = self.preprocess_image(dolp_np_raw, self.scale, target_size=(target_w, target_h), 
                                                  is_dolp_channel=True, original_filename_for_debug=str(sample_data_paths['dolp_path']))
        mask_processed_np = self.preprocess_mask(mask_pil, self.scale, target_size=(target_w, target_h),
                                                 original_filename_for_debug=str(sample_data_paths['mask_path']))

        rgb_tensor = torch.as_tensor(rgb_processed_np.copy()).contiguous()
        dolp_tensor = torch.as_tensor(dolp_processed_np.copy()).contiguous() # Should be [1, H, W] float 0-1
        mask_tensor = torch.as_tensor(mask_processed_np.copy()).long().contiguous()

        combined_input_tensor = torch.cat((rgb_tensor, dolp_tensor), dim=0)

        return_dict = { 
            'image': combined_input_tensor, 
            'mask': mask_tensor,
            'id': sample_data_paths.get('id', 'N/A_ID'), 
            'rgb_path': str(sample_data_paths.get('rgb_path', 'N/A_PATH')) 
        }
        return return_dict

# --- Test Script ---
if __name__ == '__main__':
    # (Keep the test script from the previous version, it should work with these changes)
    # ... (Make sure it calls MoistureDataset without rgb_pattern) ...
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if not hasattr(Image, 'Resampling'):
        Image.Resampling = Image 
        logging.info("Using Pillow < 9.1.0 compatibility for Resampling attributes.")
    else:
        logging.info("Using Pillow >= 9.1.0 with Image.Resampling attributes.")

    try:
        script_dir = Path(__file__).parent.resolve() 
        pytorch_unet_root = script_dir.parent         
        data_root_from_script = pytorch_unet_root / 'data'
        if not data_root_from_script.is_dir():
            if (Path.cwd() / 'Pytorch-UNet' / 'data').is_dir(): 
                data_root_from_script = Path.cwd() / 'Pytorch-UNet' / 'data'
            elif (Path.cwd() / 'data').is_dir(): 
                data_root_from_script = Path.cwd() / 'data'
            else: 
                 data_root_from_script = Path('./Pytorch-UNet/data') 
                 if not data_root_from_script.is_dir():
                     data_root_from_script = Path('./data')
        if not data_root_from_script.is_dir():
            logging.error(f"Could not automatically determine data directory. Looked for {data_root_from_script}. Please check paths.")
            sys.exit(1)
    except NameError: 
        logging.warning("__file__ not defined. Assuming CWD is shallowCNN for data paths.")
        data_root_from_script = Path('./Pytorch-UNet/data')
        if not data_root_from_script.is_dir():
            logging.error(f"Data directory {data_root_from_script} not found from CWD. Please adjust paths.")
            sys.exit(1)

    train_data_dir = data_root_from_script / 'train'
    val_data_dir = data_root_from_script / 'val'
    
    for d_dir, d_name in zip([train_data_dir, val_data_dir], ["TRAINING", "VALIDATION"]):
        resolved_dir = d_dir.resolve()
        logging.info(f"\n--- Testing data loading for {d_name} directory: {resolved_dir} ---")
        if not d_dir.is_dir():
            logging.error(f"{d_name} directory {resolved_dir} does not exist. Please check the path.")
            continue
        try:
            # Note: MoistureDataset __init__ no longer takes rgb_pattern
            dataset = MoistureDataset(data_dir=str(d_dir), scale=1.0) 
            
            if len(dataset) == 0:
                logging.error(f"{d_name} Dataset is empty!")
            else:
                logging.info(f"{d_name} Dataset created successfully with {len(dataset)} samples.")
                
                for i in range(min(len(dataset), 3)): 
                    try:
                        sample = dataset[i]
                        expected_keys = ['image', 'mask', 'id', 'rgb_path']
                        missing_keys = [key for key in expected_keys if key not in sample]
                        if missing_keys:
                            logging.error(f"{d_name} Sample {i} (Item ID: {sample.get('id', 'N/A')}, RGB: {sample.get('rgb_path', 'N/A')}) is missing keys: {missing_keys}! Found keys: {list(sample.keys())}")
                            continue

                        logging.info(f"{d_name} Sample {i} (ID: {sample['id']}) - RGB Path: {sample['rgb_path']}")
                        logging.info(f"  Combined Input Image Shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
                        logging.info(f"  Mask Shape: {sample['mask'].shape}, dtype: {sample['mask'].dtype}")
                        logging.info(f"  Unique values in mask: {torch.unique(sample['mask'])}")
                        for ch_idx in range(sample['image'].shape[0]):
                            ch_data = sample['image'][ch_idx]
                            logging.info(f"  Input Channel {ch_idx} min: {ch_data.min():.4f}, max: {ch_data.max():.4f}, mean: {ch_data.mean():.4f}")
                        
                        if i == 0 and "matplotlib" in sys.modules:
                            try:
                                import matplotlib.pyplot as plt
                                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                                fig.suptitle(f"{d_name} Sample ID: {sample['id']} | RGB: {Path(str(sample['rgb_path'])).name}")
                                
                                input_image_for_plot = sample['image'].cpu().numpy()
                                if input_image_for_plot.shape[0] == 4: 
                                    rgb_display = input_image_for_plot[:3].transpose(1, 2, 0)
                                    axs[0].imshow(np.clip(rgb_display, 0, 1)) 
                                    axs[0].set_title('Sample Input (RGB part)')
                                    axs[1].imshow(input_image_for_plot[3], cmap='viridis', vmin=0, vmax=1) # Use viridis for continuous DoLP
                                    axs[1].set_title('Sample Input (DoLP part)')
                                elif input_image_for_plot.shape[0] >= 1 :
                                     axs[0].imshow(input_image_for_plot[0], cmap='gray', vmin=0, vmax=1)
                                     axs[0].set_title(f'Sample Input (1st chan of {input_image_for_plot.shape[0]})')
                                     if input_image_for_plot.shape[0] > 1:
                                         axs[1].imshow(input_image_for_plot[1], cmap='viridis', vmin=0, vmax=1)
                                         axs[1].set_title(f'Sample Input (2nd chan of {input_image_for_plot.shape[0]})')
                                     else:
                                         axs[1].text(0.5, 0.5, 'Only 1 input channel', ha='center', va='center')
                                else:
                                     axs[0].text(0.5, 0.5, 'No image data to display', ha='center', va='center')
                                     axs[1].text(0.5, 0.5, 'No image data to display', ha='center', va='center')

                                axs[2].imshow(sample['mask'].cpu().numpy().squeeze(), cmap='gray')
                                axs[2].set_title('Sample Mask')
                                
                                save_name = f"{d_name.lower()}_dataset_sample_check_continuousDoLP_{Path(str(sample['rgb_path'])).stem}.png"
                                plt.savefig(save_name) 
                                logging.info(f"Saved sample visualization to {save_name}")
                                plt.close(fig)
                            except ImportError:
                                logging.info("Matplotlib not found, skipping visualization.")
                            except Exception as e_plt:
                                logging.warning(f"Could not plot {d_name} sample {i} due to: {e_plt}")
                    except Exception as e_sample:
                        logging.error(f"Error processing {d_name} sample {i} (ID: {dataset.ids_data[i].get('id','UNKNOWN')} RGB: {dataset.ids_data[i].get('rgb_path','UNKNOWN')}): {e_sample}", exc_info=True)
        except Exception as e:
            logging.error(f"Error during {d_name} dataset test: {e}", exc_info=True)