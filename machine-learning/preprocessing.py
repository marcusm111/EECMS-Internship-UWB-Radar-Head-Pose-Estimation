import os
import shutil
import math

def build_dataset(data_path):
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("data", "Down"), exist_ok=True)
    os.makedirs(os.path.join("data", "No Movement"), exist_ok=True)
    os.makedirs(os.path.join("data", "Right90"), exist_ok=True)
    os.makedirs(os.path.join("data", "Left"), exist_ok=True)
    os.makedirs(os.path.join("data", "Up"), exist_ok=True)
    for sub_directory in [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]:
        sub_dir_path = os.path.join(data_path, sub_directory)
        for file in [f for f in os.listdir(sub_dir_path) if os.path.isfile(os.path.join(sub_dir_path, f))]:
            if file.endswith("txt") and "BROKEN" not in file:
                if "Down" in file:
                    shutil.copy(os.path.join(data_path, sub_directory, file), os.path.join("data", "Down", file))
                elif "No Movement" in file:
                    shutil.copy(os.path.join(data_path, sub_directory, file), os.path.join("data", "No _Movement", file))
                elif "Right" in file:
                    shutil.copy(os.path.join(data_path, sub_directory, file), os.path.join("data", "Right", file))
                elif "Left" in file:
                    shutil.copy(os.path.join(data_path, sub_directory, file), os.path.join("data", "Left", file))
                elif "Up" in file:
                    shutil.copy(os.path.join(data_path, sub_directory, file), os.path.join("data", "Up", file))


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

    for root, dirs, files in os.walk(directory):
        for file in files:
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
        print("✅ All files are clean - no NaNs, infs, or invalid values found")
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

# Usage example:
main_directory = "data"
for sub_directory in os.listdir(main_directory):
    check_directory_for_nans(os.path.join(main_directory, sub_directory))

cleaned_data = "clean_data"

for sub_directory in os.listdir(main_directory):
    clean_infinities_and_save(os.path.join(main_directory, sub_directory), os.path.join(cleaned_data, sub_directory), delimiter=",")