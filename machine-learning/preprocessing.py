import os
import shutil
import math
import torch

def torch_genfromtxt(file_path, delimiter=",", dtype=torch.float32):
    """
    Load a matrix from a text file into a PyTorch tensor.
    Mimics basic functionality of `np.genfromtxt`.
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            # Split line into elements and convert to floats
            elements = line.strip().split(delimiter)
            row = [float(e) for e in elements if e]  # Skip empty strings
            data.append(row)
    # Convert to tensor and handle ragged rows (if necessary)
    tensor = torch.tensor(data, dtype=dtype)
    return tensor

def build_data_directory_structure(data_path):
    # Build data structure
    os.makedirs(os.path.join(data_path, "training_data"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "test_data"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "training_data", "old_data"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "training_data", "raw_data"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "training_data", "clean_data"), exist_ok=True)

def build_dataset_structure(dir):
    # Make class directories
    os.makedirs(os.path.join(dir, "Down"), exist_ok=True)
    os.makedirs(os.path.join(dir, "No_Movement"), exist_ok=True)
    os.makedirs(os.path.join(dir, "Right30"), exist_ok=True)
    os.makedirs(os.path.join(dir, "Right60"), exist_ok=True)
    os.makedirs(os.path.join(dir, "Right90"), exist_ok=True)
    os.makedirs(os.path.join(dir, "Left90"), exist_ok=True)
    os.makedirs(os.path.join(dir, "Left30"), exist_ok=True)
    os.makedirs(os.path.join(dir, "Left60"), exist_ok=True)
    os.makedirs(os.path.join(dir, "Up"), exist_ok=True)

def build_dataset(clean_dir, raw_dir):
    # For each class
    for sub_directory in [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]:
        sub_dir_path = os.path.join(raw_dir, sub_directory)
        # For sample in class
        for file in [f for f in os.listdir(sub_dir_path) if os.path.isfile(os.path.join(sub_dir_path, f))]:
            if file.endswith("txt"):
                shutil.copy(os.path.join(raw_dir, sub_directory, file), os.path.join(clean_dir, sub_directory, file))


def save_combined_tensors(data_path):
    tensors_path = os.path.join(data_path, "tensor_data")
    os.makedirs(tensors_path, exist_ok=True)
    build_dataset_structure(tensors_path)
    clean_data_path = os.path.join(data_path, "clean_data")
    
    for class_dir in  os.listdir(clean_data_path):
        class_full_dir = os.path.join(clean_data_path, class_dir)
        measurements_list = os.listdir(class_full_dir)
        for measurement in measurements_list:
            if measurement.endswith("txt") and "sF" in measurement:
                measurement_list = list(measurement)
                # Create left name
                measurement_list[1] = "L"
                left_corresponding_measurement = "".join(measurement_list)
                # Create right name
                measurement_list[1] = "R"
                right_corresponding_measurement = "".join(measurement_list)
                # Create paths
                measurement_path = os.path.join(class_full_dir, measurement)
                left_measurement_path = os.path.join(class_full_dir, left_corresponding_measurement)
                right_measurement_path = os.path.join(class_full_dir, right_corresponding_measurement)
                if os.path.exists(left_measurement_path) and os.path.exists(right_measurement_path): 
                    tensor_front = torch_genfromtxt(measurement_path)
                    tensor_left = torch_genfromtxt(left_measurement_path)
                    tensor_right = torch_genfromtxt(right_measurement_path)
                    combined_tensor = torch.stack([tensor_front, tensor_left, tensor_right], 0)
                    tag = measurement[2:-4]
                    current_tensor_path = os.path.join(tensors_path, class_dir, tag + ".pt")
                    torch.save(combined_tensor, current_tensor_path)

def verify_training_directory(data_path, rebuild_tensors=False):
    # Input should be data/training_data
    clean_dir_path = os.path.join(data_path, "clean_data")
    raw_dir_path = os.path.join(data_path, "raw_data")
    clean_dir = os.listdir(clean_dir_path)
    raw_dir = os.listdir(raw_dir_path)


    if len(clean_dir) == 0:
        build_dataset_structure(clean_dir_path)

    # Check if the subdirectories are empty
    if len([d for d in clean_dir if len(os.listdir(os.path.join(clean_dir_path, d))) > 0 ]) == 0:
        # Training directory empty, check for raw data
        if len(raw_dir) == 0:
            # No training or raw data
            print("No training or raw data, please put raw data in raw_data")
            return False
        
        # Clean nans
        # Re-initialise clean_dir list
        raw_dir = os.listdir(raw_dir_path)
        clean_infinities_and_save(raw_dir_path, clean_dir_path)

    # Check each sample has a matching from the other sensor
    clean_dir = os.listdir(clean_dir_path)
    complete_match = True
    for class_dir in clean_dir:
        class_full_dir = os.path.join(clean_dir_path, class_dir)
        measurements_list = os.listdir(class_full_dir)
        num_measurements = len(measurements_list)
        total_num_samples = num_measurements // 3
        num_samples = 0
        for measurement in measurements_list:
            if measurement.endswith("txt") and "sF"in measurement:
                measurement_list = list(measurement)
                measurement_list[1] = "R"
                right_measurement = "".join(measurement_list)
                measurement_list[1] = "L"
                left_measurement = "".join(measurement_list)

                if (left_measurement in measurements_list) and (right_measurement in measurements_list):
                    num_samples += 1
                else:
                    print(f"No corresponding measurement to: {measurement}")

        if num_samples != total_num_samples:
            print(f"Invalid measurement alignment in class: {class_dir}")
            complete_match = False

        print(f"Num samples: {num_samples}, num_measurements: {num_measurements}")

    if complete_match and (not os.path.exists(os.path.join(data_path, "tensor_data")) or rebuild_tensors):
        save_combined_tensors(data_path)


def check_directory_for_nans(directory, delimiter=","):
    """
    Checks all .txt files in a directory for:
    - NaN values
    - Infinity values
    - Non-numeric values
    
    Args:
        directory (str): Path to directory containing .txt files
        delimiter (str): Column delimiter used in the files
    """
    problematic_files = []
    files_checked = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            files_checked += 1
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                has_nan = False
                has_inf = False
                has_invalid = False

                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        elements = line.strip().split(delimiter)
                        for col_num, element in enumerate(elements, 1):
                            element = element.strip()
                            if not element:
                                continue  # Skip empty elements
                            
                            try:
                                val = float(element)
                            except ValueError:
                                print(f"⚠️ Invalid value '{element}' in {file_path}")
                                print(f"  → Line {line_num}, Column {col_num}")
                                has_invalid = True
                                continue
                            
                            if math.isnan(val):
                                print(f"🚫 NaN detected in {file_path}")
                                print(f"  → Line {line_num}, Column {col_num}")
                                has_nan = True
                            elif math.isinf(val):
                                print(f"🚫 Infinity detected in {file_path}")
                                print(f"  → Line {line_num}, Column {col_num}")
                                has_inf = True

                # Record files with any issues
                if has_nan or has_inf or has_invalid:
                    problematic_files.append({
                        'path': file_path,
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'has_invalid': has_invalid
                    })

    # Print summary
    print("\n=== Summary ===")
    if not problematic_files:
        print(f"✅ All {files_checked} files are clean - no NaNs, infs, or invalid values found")
    else:
        print(f"❌ Found issues in {len(problematic_files)} files:")
        for file_info in problematic_files:
            issues = []
            if file_info['has_nan']: issues.append("NaNs")
            if file_info['has_inf']: issues.append("Infs")
            if file_info['has_invalid']: issues.append("Invalid values")
            print(f"  - {file_info['path']}: {', '.join(issues)}")

def clean_infinities_and_save(input_dir, output_dir, delimiter=","):
    """
    Processes all .txt files in input_dir, replaces inf/-inf with 0,
    handles NaNs and invalid entries, and saves cleaned files to output_dir.
    Preserves original directory structure and reports cleaning statistics.
    """
    # Initialize counters
    total_files = 0
    total_inf = 0
    total_nan = 0
    total_invalid = 0

    for root, dirs, files in os.walk(input_dir):
        # Create mirror directory in output
        rel_path = os.path.relpath(root, input_dir)
        output_root = os.path.join(output_dir, rel_path)
        os.makedirs(output_root, exist_ok=True)

        for file in files:
            if file.endswith(".txt"):
                total_files += 1
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_root, file)
                
                file_inf = 0
                file_nan = 0
                file_invalid = 0

                with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
                    for line in fin:
                        cleaned = []
                        elements = line.strip().split(delimiter)
                        
                        for elem in elements:
                            elem = elem.strip()
                            if not elem:
                                cleaned.append("0")  # Replace empty with 0
                                continue
                                
                            try:
                                val = float(elem)
                                if math.isinf(val):
                                    cleaned.append("0")
                                    file_inf += 1
                                elif math.isnan(val):
                                    cleaned.append("0")
                                    file_nan += 1
                                else:
                                    cleaned.append(elem)  # Keep valid numbers
                            except ValueError:
                                cleaned.append("0")
                                file_invalid += 1
                                
                        fout.write(delimiter.join(cleaned) + "\n")

                # Update totals
                total_inf += file_inf
                total_nan += file_nan
                total_invalid += file_invalid

                # Print file summary if any issues
                if file_inf + file_nan + file_invalid > 0:
                    print(f"Cleaned: {input_path}")
                    print(f"  → Infinities: {file_inf}")
                    print(f"  → NaNs: {file_nan}")
                    print(f"  → Invalid entries: {file_invalid}")

    # Final report
    print("\n=== Cleaning Complete ===")
    print(f"Processed files: {total_files}")
    print(f"Total infinities replaced: {total_inf}")
    print(f"Total NaNs replaced: {total_nan}")
    print(f"Total invalid entries replaced: {total_invalid}")
    print(f"\nCleaned files saved to: {output_dir}")
