"""
This script contains file-format functions.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
from pathlib import Path


def txt_to_csv(
    route: str | Path,
    sep: Optional[str] = '|',
    output_path: Optional[str | Path] = None,
    index: bool = False
) -> pd.DataFrame:
    """
    Converts a delimited .txt file to a CSV file using the first line as column headers.

    Returns
    -------
    pd.DataFrame
        The resulting DataFrame parsed from the .txt file.
    
    Notes
    -----
    - Assumes the first line in the .txt file contains the column names.
    - Automatically disables index in the CSV output.
    - Handles both string and Path inputs for file paths.
    """
    df = pd.read_csv(route, sep=sep)

    if output_path is None:
        output_path = Path(route).with_suffix('.csv')

    df.to_csv(output_path, index=index)

    return df

 

    