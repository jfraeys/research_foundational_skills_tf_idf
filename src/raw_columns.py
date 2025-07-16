import io
import os

import msoffcrypto
import openpyxl

decrypted_workbook = io.BytesIO()


def get_secret(file_name="PASSWD_FILE", secrets_dir="secrets"):
    # Construct the path to the secrets directory and the specific file
    file_path = os.path.join(secrets_dir, file_name)

    # Read the secret from the file once, using a context manager
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Decrypt the workbook and load the data in one step
password = get_secret("COOP_FULLTIME_PASSWD")
if password:
    with open("data/raw/Co-op_FullTime.xlsx", "rb") as f:
        office_file = msoffcrypto.OfficeFile(f)
        office_file.load_key(password=password)
        office_file.decrypt(decrypted_workbook)

    workbook = openpyxl.load_workbook(decrypted_workbook, read_only=True)

    columns = next(workbook.active.values)

    print(columns)
