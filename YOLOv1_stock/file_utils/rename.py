import os


def rename_files_in_folder(folder_path):
    """
    Rename all files in the given folder to sequential numbers while keeping their extensions.
    """
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Sort files to maintain any existing order, optional
    files.sort()

    # Start renaming files
    for i, filename in enumerate(files, start=1):
        # Split the filename into name and extension
        basename, extension = os.path.splitext(filename)
        # Construct the new filename with the same extension
        new_filename = f"{i}{extension}"
        # Construct the full old and new file paths
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed {old_file} to {new_file}")


# Example usage
folder_path = "../dataset_midterm/test_images/stock_test_images"  # Replace with your folder path
rename_files_in_folder(folder_path)
