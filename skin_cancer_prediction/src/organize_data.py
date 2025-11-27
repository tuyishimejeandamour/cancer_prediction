import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

def organize_ham10000(csv_path, images_dirs, base_output_dir):
    """
    Organizes HAM10000 dataset into train/test folders with benign/malignant subfolders.
    
    HAM10000 Classes:
    nv       Melanocytic nevi (Benign)
    mel      Melanoma (Malignant)
    bkl      Benign keratosis-like lesions (Benign)
    bcc      Basal cell carcinoma (Malignant)
    akiec    Actinic keratoses (Malignant)
    vasc     Vascular lesions (Benign)
    df       Dermatofibroma (Benign)
    """
    
    # Define mappings
    malignant_classes = ['mel', 'bcc', 'akiec']
    benign_classes = ['nv', 'bkl', 'vasc', 'df']
    
    print("Reading CSV...")
    df = pd.read_csv(csv_path)
    
    # Add binary label
    df['binary_label'] = df['dx'].apply(lambda x: 'malignant' if x in malignant_classes else 'benign')
    
    print(f"\nDataset Statistics:")
    print(f"Total images: {len(df)}")
    print(f"Benign: {len(df[df['binary_label'] == 'benign'])}")
    print(f"Malignant: {len(df[df['binary_label'] == 'malignant'])}")
    print(f"\nClass distribution:")
    print(df['dx'].value_counts())
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['binary_label'], random_state=42)
    
    print(f"\nTrain set: {len(train_df)} images")
    print(f"Test set: {len(test_df)} images")
    
    def find_image(img_id, search_dirs):
        """Search for image in multiple directories"""
        for directory in search_dirs:
            img_path = os.path.join(directory, f"{img_id}.jpg")
            if os.path.exists(img_path):
                return img_path
        return None
    
    def copy_images(dataframe, subset_name):
        print(f"\nProcessing {subset_name} set...")
        copied = 0
        not_found = 0
        
        for _, row in dataframe.iterrows():
            img_id = row['image_id']
            label = row['binary_label']
            
            # Search for image in all provided directories
            src_path = find_image(img_id, images_dirs)
            
            if src_path:
                dst_dir = os.path.join(base_output_dir, subset_name, label)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, f"{img_id}.jpg")
                shutil.copy2(src_path, dst_path)
                copied += 1
            else:
                not_found += 1
                
        print(f"Copied: {copied}, Not found: {not_found}")

    copy_images(train_df, 'train')
    copy_images(test_df, 'test')
    print("\n Organization complete!")
    print(f"\nOutput structure:")
    print(f"  {base_output_dir}/train/benign/")
    print(f"  {base_output_dir}/train/malignant/")
    print(f"  {base_output_dir}/test/benign/")
    print(f"  {base_output_dir}/test/malignant/")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Update paths to match your actual data location
    csv_file = os.path.join(base_path, 'data', 'HAM10000_metadata.csv')
    
    # List of directories containing images (both part 1 and part 2)
    img_folders = [
        os.path.join(base_path, 'data', 'HAM10000_images_part_1'),
        os.path.join(base_path, 'data', 'HAM10000_images_part_2')
    ]
    
    output_dir = os.path.join(base_path, 'data')
    
    if os.path.exists(csv_file) and all(os.path.exists(f) for f in img_folders):
        organize_ham10000(csv_file, img_folders, output_dir)
    else:
        print("ERROR: HAM10000 dataset not found!")
        print(f"Expected CSV at: {csv_file} - {' Found' if os.path.exists(csv_file) else '✗ Not found'}")
        for folder in img_folders:
            print(f"Expected Images at: {folder} - {' Found' if os.path.exists(folder) else '✗ Not found'}")
