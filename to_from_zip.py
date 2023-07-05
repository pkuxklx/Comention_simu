import os
import zipfile, shutil

def to_from_zip():
    option = str(input('convert to or from zip. (t/f)? '))
    
    # print("Input 'q' to quit.")

    if option == 'f':
        zip_name = str(input("the .zip file's name: ")) # xxx.zip
        extract_path = str(input('path to extract to (if None, extract to the current working directory): '))

        with zipfile.ZipFile(file = zip_name, mode = 'r') as files:
            files.extractall(path = extract_path)

    elif option == 't':
        dir_name = str(input("the directory's name:")) # a directory, e.g. "./Co_simu/data"
        base_zip_name = str(input("the name of file to create (without .zip suffix): "))

        shutil.make_archive(base_name = base_zip_name, format = 'zip', root_dir = dir_name)
    else:
        raise ValueError("Option error.")

if __name__ == '__main__':
    print(f'current working directory: {os.getcwd()}')
    to_from_zip()
    print('successfully finished.')