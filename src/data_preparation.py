import pandas as pd

def clean_spotify_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove irrelevant columns and missing values"""
    
    columns_to_drop = [
       "Unnamed: 0.1",
        "Unnamed: 0",
        "track_id",
        "artists",
        "album_name",
        "track_name"
    ]
    
    df = df.drop(columns=columns_to_drop)
    
    df = df.dropna().reset_index(drop=True)
    
    return df