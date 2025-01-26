import os
import clearml as cml

from dotenv import find_dotenv, load_dotenv


def load_explicit_clearml_settings_from_dotenv(env_file_path: str = None, is_strict: bool = True, logger = None):
    """
    Loads environment variables from specified .env file (if None, it will be found
    automatically) and shows CLEARML_WEB_HOST variable state.
    Useful when working with multiple ClearML servers to explicitly
    specify what server and creds to use.

    Args:
        env_file_path (str, optional): Path to .env file. Defaults to None.
        is_strict (bool, optional): Whether to check if ClearML settings are present and raise exception if not
    """
    # Display old state
    msg = f"Old {os.getenv('CLEARML_WEB_HOST')=}"
    logger.debug(msg) if logger else print(msg)

    # Find the env file if not specified
    if env_file_path is None:
        env_file_path = find_dotenv()

    # Load environment variables from .env file, checking if anything is loaded
    assert load_dotenv(env_file_path, override=True)
    msg = f"Loaded .env file from: {env_file_path}"
    logger.info(msg) if logger else print(msg)

    # Display the new state
    msg = f"New {os.getenv('CLEARML_WEB_HOST')=}"
    logger.debug(msg) if logger else msg
    if is_strict:
        assert os.getenv('CLEARML_WEB_HOST') is not None, f"The ClearML variables seems to be missed in the .env file"


def get_cml_task(task_name):
    task = cml.Task.init(
        project_name="AI-OCR",
        task_name=task_name,
        auto_resource_monitoring=False,
        continue_last_task=False
    )
    return task
