import pandas as pd
from pathlib import Path

def get_data(transform: bool, data_dir: str = None) -> pd.DataFrame:
    """
    Load dataset for analysis.

    transform : bool
        If True, load Box-Cox transformed data; otherwise, raw data.
    data_dir : str, optional
        Directory containing the data. Defaults to data_analysis relative to the project root.

    Returns
    -------
    pd.DataFrame
    """

    base_path = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data_analysis"
    filename = "Boxcox_keyData.csv" if transform else "keyData.csv"
    filepath = base_path / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    return pd.read_csv(filepath, index_col=0)