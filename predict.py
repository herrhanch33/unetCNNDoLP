import sys
import argparse
import logging
import os
import re # For parsing filenames

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont # Added ImageDraw, ImageFont
# from torchvision import transforms # Not strictly needed by this version

from unet import UNet
from utils.utils import plot_img_and_mask # Assuming this is modified to take prediction_text
from pathlib import Path 

def prepare_input_tensor(rgb_pil_img, dolp_npy_array, scale_factor, device):
    # This function should correctly handle continuous DoLP (0-1 float)
    # and resize both RGB and DoLP to consistent scaled dimensions.
    
    # --- Preprocess RGB ---
    w_rgb, h_rgb = rgb_pil_img.size
    newW_rgb, newH_rgb = int(scale_factor * w_rgb), int(scale_factor * h_rgb)
    assert newW_rgb > 0 and newH_rgb > 0, 'Scale is too small for RGB image'
    
    rgb_pil_resized = rgb_pil_img.resize((newW_rgb, newH_rgb), resample=Image.Resampling.BICUBIC)
    rgb_np = np.asarray(rgb_pil_resized)

    if rgb_np.ndim == 2: 
        rgb_np = rgb_np[np.newaxis, ...]
    else: 
        rgb_np = rgb_np.transpose((2, 0, 1)) 

    if (rgb_np > 1).any(): 
        rgb_np = rgb_np / 255.0
    rgb_tensor = torch.from_numpy(rgb_np.astype(np.float32))

    # --- Preprocess DoLP (Continuous) ---
    dolp_np_raw = dolp_npy_array.astype(np.float32) 
    if dolp_np_raw.ndim == 3: dolp_np_raw = dolp_np_raw.squeeze()
    assert dolp_np_raw.ndim == 2, "DoLP numpy array must be 2D after squeeze"
    
    dolp_np_raw = np.clip(dolp_np_raw, 0.0, 1.0) 

    # Convert to PIL for resizing using BICUBIC
    temp_pil_for_resize = Image.fromarray((dolp_np_raw * 255).astype(np.uint8))
    # Resize DoLP to match RGB's scaled dimensions
    dolp_pil_resized = temp_pil_for_resize.resize((newW_rgb, newH_rgb), resample=Image.Resampling.BICUBIC)
    dolp_np_resized = np.asarray(dolp_pil_resized).astype(np.float32) / 255.0
    dolp_np_resized = np.clip(dolp_np_resized, 0.0, 1.0) 
        
    dolp_tensor = torch.from_numpy(dolp_np_resized[np.newaxis, ...]) 

    # No need for the extra check if rgb_tensor.shape[1:] != dolp_tensor.shape[1:]:
    # because we are now explicitly resizing DoLP to newW_rgb, newH_rgb.

    combined_tensor = torch.cat((rgb_tensor, dolp_tensor), dim=0) 
    return combined_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)


def predict_img(net,
                input_tensor, 
                full_img_size, 
                out_threshold=0.5):
    net.eval()
    with torch.no_grad():
        output_logits = net(input_tensor) # Get raw logits from the model
        output_probs = torch.sigmoid(output_logits).cpu() # Apply sigmoid to get probabilities [0,1]
        
        # Upscale probabilities to original image size
        output_resized_probs = F.interpolate(output_probs, (full_img_size[1], full_img_size[0]), mode='bilinear', align_corners=False)
        
        if net.n_classes > 1: # Should not happen
            mask = output_resized_probs.argmax(dim=1)
        else: # Binary case
            mask = (output_resized_probs > out_threshold) # Threshold probabilities

    # Return both the binary mask (for saving/visualization) and the probability map (for more info)
    return mask[0].long().squeeze().numpy(), output_resized_probs[0].squeeze().cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT_RGB_IMAGE', nargs='+', 
                        help='Filenames of input RGB images', required=True)
    parser.add_argument('--dolp_dir', type=str, default=None, 
                        help='Directory for DoLP .npy files. If None, attempts to derive from RGB image path.')
    # ### REMOVED rgb_pattern argument ###
    # parser.add_argument('--rgb_pattern', type=str, default='_I90_', ...)

    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.25, 
                        help='Scale factor for the input images before feeding to network (0 to 1)')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes (1 for binary)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling in UNet')
    
    return parser.parse_args()

def get_output_filenames(args):
    def _generate_name(fn):
        base = Path(fn).stem
        return f'predictions/{base}_PRED.png'
    if not args.no_save and not os.path.exists('predictions'):
        os.makedirs('predictions')
    return args.output or list(map(_generate_name, args.input))

def mask_to_image(mask: np.ndarray, mask_values=None): 
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    out_mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(out_mask)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Pillow Resampling attributes check
    if not hasattr(Image, 'Resampling'):
        Image.Resampling = Image 
        logging.info("Using Pillow < 9.1.0 compatibility for Resampling attributes.")
    else:
        logging.info("Using Pillow >= 9.1.0 with Image.Resampling attributes.")

    in_files_rgb = args.input 
    out_files = get_output_filenames(args)

    net = UNet(n_channels=4, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    try:
        state_dict = torch.load(args.model, map_location=device)
        if 'mask_values' in state_dict: 
            mask_values_from_checkpoint = state_dict.pop('mask_values', [0, 1]) 
        else: 
            mask_values_from_checkpoint = [0, 1] # For binary: background, foreground
        net.load_state_dict(state_dict)
        logging.info('Model loaded!')
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        if "Error(s) in loading state_dict for UNet" in str(e):
            logging.error("This often means a mismatch in model architecture (n_channels, n_classes, or bilinear flag) "
                          "between the saved model and the one defined here.")
            logging.error(f"Current script UNet bilinear: {args.bilinear}. Ensure this matches training.")
        sys.exit(1)

    for i, rgb_filename in enumerate(in_files_rgb):
        logging.info(f'\nPredicting image {rgb_filename} ...')
        try:
            rgb_pil_img = Image.open(rgb_filename).convert('RGB')
            original_rgb_size = rgb_pil_img.size 

            # --- Derive DoLP filename (Robust Logic) ---
            rgb_path_obj = Path(rgb_filename)
            stem_name = rgb_path_obj.stem
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
                    logging.warning(f"Could not parse base_identifier from RGB filename '{stem_name}' for DoLP/Mask lookup. Skipping '{rgb_filename}'.")
                    continue
            
            if args.dolp_dir:
                dolp_dir_path = Path(args.dolp_dir)
            else:
                rgb_parent_dir = rgb_path_obj.parent 
                dataset_subset_dir = rgb_parent_dir.parent 
                dolp_dir_path = dataset_subset_dir / 'dolp_images'
                if not dolp_dir_path.is_dir():
                    logging.warning(f"Guessed DoLP directory {dolp_dir_path} not found. Defaulting to RGB image directory. Consider using --dolp_dir.")
                    dolp_dir_path = rgb_path_obj.parent 
            
            dolp_filename = dolp_dir_path / f"DoLP_raw_{base_identifier}.npy"

            if not dolp_filename.exists():
                logging.error(f"DoLP file not found: {dolp_filename} (derived from RGB {rgb_filename}). Skipping.")
                continue
            
            dolp_npy_array = np.load(dolp_filename)

            input_tensor = prepare_input_tensor(rgb_pil_img, dolp_npy_array, args.scale, device)
            
            # predict_img now returns (binary_mask_np, probability_map_np)
            binary_mask_np, prob_map_np = predict_img(net=net,
                                                      input_tensor=input_tensor,
                                                      full_img_size=original_rgb_size,
                                                      out_threshold=args.mask_threshold)

            # --- Determine water presence text based on the binary mask ---
            water_detected_text = "Possibility of Water: No"
            water_pixel_count = 0
            water_percentage = 0.0

            if np.any(binary_mask_np == 1): # If any pixel is predicted as water
                water_pixel_count = np.sum(binary_mask_np == 1)
                total_pixels = binary_mask_np.size
                water_percentage = (water_pixel_count / total_pixels) * 100
                
                if water_percentage > 1.0: # Example: more than 1% of image is water
                    water_detected_text = f"High Possibility of Water ({water_percentage:.2f}%)"
                elif water_percentage > 0.01: # Example: some water detected
                    water_detected_text = f"Possibility of Water: Yes ({water_percentage:.2f}%)"
                else:
                    water_detected_text = f"Possibility of Water: Trace ({water_percentage:.2f}%)"
            
            logging.info(water_detected_text)


            if not args.no_save:
                out_filename = out_files[i]
                Path(out_filename).parent.mkdir(parents=True, exist_ok=True)
                
                # Save the binary mask image
                result_img_pil = mask_to_image(binary_mask_np, mask_values_from_checkpoint) 
                
                # Add text to the saved image (optional, can make image busy)
                draw = ImageDraw.Draw(result_img_pil)
                try:
                    font = ImageFont.truetype("arial.ttf", 15) # Try to load a common font
                except IOError:
                    font = ImageFont.load_default() # Fallback
                text_lines = water_detected_text.split('\n')
                y_offset = 10
                for line in text_lines:
                    draw.text((10, y_offset), line, font=font, fill=128) # Gray text
                    y_offset += (font.getbbox("A")[3] - font.getbbox("A")[1] + 2) if hasattr(font, 'getbbox') else 12


                result_img_pil.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if args.viz:
                logging.info(f'Visualizing results for image {rgb_filename}, close plot to continue...')
                # Assuming your plot_img_and_mask is modified to accept prediction_text
                plot_img_and_mask(rgb_pil_img, binary_mask_np, prediction_text=water_detected_text) 

        except Exception as e:
            logging.error(f"Error processing file {rgb_filename}: {e}", exc_info=True)