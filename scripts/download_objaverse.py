import os
import time
from tqdm import tqdm

output_dir = "./dataset/objaverse"
os.makedirs(output_dir, exist_ok=True)
os.environ["OBJAVERSE_SAVE_PATH"] = output_dir

import objaverse

def main():
    print("Loading LVIS annotations (approx 46k objects)...")
    lvis = objaverse.load_lvis_annotations()
    
    # Flatten the list of UIDs
    uids = list(set(uid for vals in lvis.values() for uid in vals))
    print(f"Total objects to download: {len(uids):,}")
    
    print(f"Downloading to {output_dir}...")
    
    batch_size = 100
    successful = 0
    
    for i in tqdm(range(0, len(uids), batch_size), desc="Downloading batches"):
        batch_uids = uids[i:i+batch_size]
        try:
            # We use 4 processes to avoid rate limiting
            objects = objaverse.load_objects(uids=batch_uids, download_processes=4)
            successful += len(objects)
        except Exception as e:
            print(f"Error downloading batch {i}: {e}")
            time.sleep(5)
            
    print(f"Successfully downloaded {successful}/{len(uids)} objects.")

if __name__ == "__main__":
    main()
