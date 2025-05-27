# In Pytorch-UNet/utils/utils.py
import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask, prediction_text=""): # Add new argument
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i, cmap='gray') # Assuming binary mask for class 1
    
    # ### NEW: Add prediction text ###
    if prediction_text:
        plt.figtext(0.5, 0.01, prediction_text, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    plt.xticks([]), plt.yticks([])
    plt.show()