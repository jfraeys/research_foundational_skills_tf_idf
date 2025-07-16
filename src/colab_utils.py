import os
from typing import Any, Dict


def is_colab() -> bool:
    """
    Check if the current environment is Google Colab.

    Returns:
        bool: True if running in Google Colab, False otherwise.
    """
    try:
        import google.colab

        return True
    except ImportError:
        return False


def mount_drive() -> str:
    """
    Mount Google Drive in Colab, if not already mounted.

    Returns:
        str: Path to the mounted drive (default is '/content/drive').
    """
    drive_mounted_path = "/content/drive"
    if not os.path.exists(drive_mounted_path):
        from google.colab import drive

        drive.mount(drive_mounted_path)
    return drive_mounted_path


def get_project_base_dir() -> str:
    """
    Determine the base directory of the project, accounting for `src/` or `notebooks/` as starting points.

    Returns:
        str: The base directory of the project.
    """
    current_dir = os.getcwd()
    while not {"notebooks", "src"}.issubset(os.listdir(current_dir)):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Root directory reached
            raise FileNotFoundError(
                "Project base directory not found. Ensure it contains 'notebooks' and 'src'."
            )
        current_dir = parent_dir
    return current_dir


def get_project_paths(base_dir: str, filenames: Dict[str, str]) -> Dict[str, str]:
    """
    Construct file paths for the project based on the base directory.

    Args:
        base_dir (str): Base directory of the project.
        filenames (dict): Dictionary of filenames with keys as identifiers and values as filenames.

    Returns:
        dict: Dictionary with full paths for each file.
    """
    return {
        key: os.path.join(base_dir, filename) for key, filename in filenames.items()
    }


def setup_environment(
    filenames: Dict[str, str], colab_project_path: str = ""
) -> Dict[str, Any]:
    """
    Set up the environment by detecting Colab, mounting Google Drive if needed,
    and ensuring the correct project directory.

    Args:
        filenames (dict): Dictionary of filenames with keys as identifiers and values as filenames.
        colab_project_path (str): Path to the project folder in Google Drive if running in Colab.

    Returns:
        dict: Dictionary containing paths and Colab status.
    """
    # Check if running in Colab
    in_colab = is_colab()
    if in_colab:
        print("Running in Colab. Mounting Google Drive...")
        drive_mounted_path = mount_drive()
        if not colab_project_path:
            raise ValueError(
                "Please provide 'colab_project_path' when running in Colab."
            )
        base_dir = os.path.join(drive_mounted_path, colab_project_path)
    else:
        # Locate the project base directory locally
        base_dir = get_project_base_dir()

    # Get project paths
    paths = get_project_paths(base_dir, filenames)
    paths["project_dir"] = base_dir
    paths["in_colab"] = in_colab
    return paths


if __name__ == "__main__":
    # Example usage
    project_filenames = {
        "job_desc_filename": "data/job_desc_filename.csv",
        "foundal_skills_filename": "data/foundal_skills_filename.csv",
    }

    try:
        env_info = setup_environment(
            filenames=project_filenames,
            colab_project_path=os.getcwd(),
        )
        if env_info["in_colab"]:
            print("Google Colab detected.")
        else:
            print("Not running in Google Colab.")
        print(f"Job Description File: {env_info['job_desc_filename']}")
        print(f"Foundational Skills File: {env_info['foundal_skills_filename']}")
    except Exception as e:
        print(f"Error setting up environment: {e}")
