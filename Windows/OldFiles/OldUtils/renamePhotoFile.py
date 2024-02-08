
import os
import glob


def get_sorted_prefixes(dir):
    subjects = []

    # Use sorted to get files ordered by name
    for filename in sorted(glob.glob(dir + '/*.*')):
        print(filename)
        fotoName = filename.split(dir + '\\')[1]
        subject = fotoName.split("_")[0]
        if subject not in subjects:
            subjects.append(subject)

    # print(len(subjects))
    return subjects


def rename_files(folder1, folder2):
    # Get prefixes in order of appearance from your function
    prefixes = get_sorted_prefixes(folder1)

    if prefixes is None:
        return

    # List of files in folder2
    files_to_rename = sorted([f for f in os.listdir(folder2)])

    # Check if there are enough prefixes to rename all the files
    if len(files_to_rename) > len(prefixes):
        print("There are not enough prefixes to rename all the files.")
        return

    # Rename the files in folder2 using the prefixes from your function
    for prefix, old_name in zip(prefixes, files_to_rename):
        new_name = f"{prefix}_{old_name}"
        old_path = os.path.join(folder2, old_name)
        new_path = os.path.join(folder2, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")


# Example of usage
folder1 = "D:\PiscopoRoberto\FER\CK+\surprise"
folder2 = "D:\PiscopoRoberto\FER\CK+PreProcessed\SURPRISE"

rename_files(folder1, folder2)




