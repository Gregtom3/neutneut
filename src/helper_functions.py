import os
from datetime import datetime
from PIL import Image
from collections import defaultdict
import tensorflow as tf

def create_output_directory(base_dir="./out"):
    # Get the current date in MM_DD_YYYY format
    date_str = datetime.now().strftime("%m_%d_%Y")
    
    # Start the counter at 0000
    counter = 0
    
    # Loop to find an available directory
    while True:
        # Format the directory name with zero-padded counter
        dir_name = f"{base_dir}/{date_str}_{counter:04}"
        
        # Check if the directory already exists
        if not os.path.exists(dir_name):
            # If it doesn't exist, create the directory and return its path
            os.makedirs(dir_name)
            return dir_name
        
        # Increment the counter
        counter += 1
        
        
def create_gif_from_pngs(outdir, gif_prefix="output", duration=500, savedir=None):
    """
    Create separate GIFs for each event from all .png files in the specified directory.
    Handles .png files ending in _train, _test, _batch, or neither, creating a GIF that includes all relevant files.

    Parameters:
    -----------
    outdir : str
        The directory containing .png files to be compiled into GIFs.
    gif_prefix : str
        The prefix for the output GIF files.
    duration : int
        Duration of each frame in the GIF in milliseconds.
    savedir: str
        Directory to save gif
    """

    if savedir == None:
        savedir = outdir
        
    # Find all .png files in the directory and group them by event number and batch number
    png_files = sorted([f for f in os.listdir(outdir) if f.endswith('.png')])

    # Dictionary to hold lists of files for each event, category, and batch
    event_files = defaultdict(lambda: {'train': [], 'test': [], 'none': [], 'batch': defaultdict(list)})

    # Group files by event number, category, and batch number
    for png_file in png_files:
        event_number = None
        batch_number = None
        category = None
        # Extract event number from the filename (assuming "ev#" is in the filename)
        for part in png_file.split('_'):
            if part.startswith("ev"):
                event_number = part
            if "train" in part:
                category = 'train'
            if "test" in part:
                category = 'test'

        # Append files to the correct category or batch group
        if event_number:
            if category == 'batch' and batch_number:
                event_files[event_number][category][batch_number].append(png_file)
            else:
                event_files[event_number][category].append(png_file)
    
    # Create a GIF for each event, category, and batch
    for event, file_dict in event_files.items():
        # Handle train, test, and none categories
        for category in ['none', 'train', 'test']:
            all_files = file_dict[category]
            if len(all_files) == 0:
                continue
            images = [Image.open(os.path.join(outdir, f)) for f in sorted(all_files)]
            gif_path = os.path.join(savedir, f"{gif_prefix}_{event}_{category}.gif")
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
            print(f"GIF saved at: {gif_path}")
        
        
def check_gpus():
    print("Tensorflow Version ==",tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus)==0:
        print("*** NO GPUS FOUND ***")
    else:
        for gpu in gpus:
            print(f"Device Name: {gpu.name}, Device Type: {gpu.device_type}")
