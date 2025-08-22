import os
import tempfile
import pandas as pd


def atomic_save_parquet(df: pd.DataFrame, final_path: str, *args, **kwargs):
    """
    Saves a DataFrame to a Parquet file atomically by writing to a temporary file first, 
    then replacing the final file using os.replace(). Ensures cleanup on failure.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        final_path (str): The destination file path.
    """

    dir_name = os.path.dirname(final_path)
    # Move one folder up from the final destination
    dir_name = os.path.join(os.path.dirname(dir_name), "temp")

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    

    # Create a temporary file in the same directory
    with tempfile.NamedTemporaryFile(dir=dir_name, suffix=".parquet", delete=False) as temp_dir:
        temp_path = temp_dir.name

    try:
        # Save to the temp Parquet file
        df.to_parquet(temp_path, *args, **kwargs)

        # Atomic replace - ensures no partial writes
        os.replace(temp_path, final_path)
        
    
    finally:
        # Ensure temp file cleanup if it still exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
