from pathlib import Path

def inspect_labels(data_dir: str):
    folder_path = Path(data_dir)
    txt_files = list(folder_path.glob('*.txt'))
    
    empty_count = 0
    populated_count = 0
    
    for txt in txt_files:
        if txt.stat().st_size == 0:
            empty_count += 1
        else:
            populated_count += 1
            
    print(f"Total .txt files checked: {len(txt_files)}")
    print(f"Empty (0 bytes / True Backgrounds): {empty_count}")
    print(f"Populated (Contains Annotations): {populated_count}")

if __name__ == "__main__":
    inspect_labels("datasets/custom_annotated_v1/obj_train_data")