import binascii

import pandas as pd
from shapely import wkb


def decode_point_geom(wkb_hex: str):
    """
    Convert WKB hex string to (longitude, latitude) tuple.
    
    WKB stands for Well-Known Binary — a binary encoding for geometries such as POINT, LINESTRING, etc.
    In this dataset, `point_geom` is a WKB-encoded POINT geometry with SRID 4326 (i.e., GPS/WGS84).
    
    The format is as follows:
        - Starts with a header:
            - Byte order: 01 for little-endian
            - Geometry type: 01000000 = POINT (WKB type 1)
            - SRID: 20E61000 = 4326 (in little-endian hex → standard GPS CRS)
        - Then two 64-bit floats (little-endian IEEE-754 format):
            - Longitude (X)
            - Latitude  (Y)
    
    Example (in hex): 
        0101000020E610000088B9835CD22E00C0D0B936AFC59C4540

        → "88B9835CD22E00C0" → -2.0228622 (longitude)
        → "D0B936AFC59C4540" → 43.22478   (latitude)
    
    This function uses `shapely.wkb.loads()` to decode the binary into a Shapely Point.
    
    Parameters:
        wkb_hex (str): Hex-encoded WKB string (e.g., "0101000020E6100000...")
    
    Returns:
        tuple: (longitude, latitude) as float values
    """
    if pd.isna(wkb_hex):
        return (None, None)

    try:
        # Convert hex string to raw binary (bytes)
        wkb_bytes = binascii.unhexlify(wkb_hex)
        
        # Decode using shapely — interprets WKB into a geometry object
        geom = wkb.loads(wkb_bytes)

        # Extract X (longitude) and Y (latitude) from the Point object
        return geom.x, geom.y
    
    except Exception as e:
        print(f"Error decoding WKB: {wkb_hex}, Error: {e}")
        return (None, None)

def add_lon_lat_columns(df: pd.DataFrame, geom_column: str = "point_geom"):
    """
    Given a DataFrame with a 'point_geom' column in WKB hex format,
    decode it into standard longitude and latitude float columns.

    Adds two new columns:
        - 'longitude_decoded': the decoded X coordinate from the POINT geometry
        - 'latitude_decoded' : the decoded Y coordinate from the POINT geometry

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the WKB hex geometries
        geom_column (str): The name of the column containing WKB hex strings (default: "point_geom")

    Returns:
        pd.DataFrame: The DataFrame with two additional columns
    """
    # Apply decoding to each row, creating a series of (lon, lat) tuples
    coords = df[geom_column].apply(decode_point_geom)

    # Split tuple into two separate columns
    df["longitude_decoded"] = coords.apply(lambda tup: tup[0])
    df["latitude_decoded"] = coords.apply(lambda tup: tup[1])
    
    return df
