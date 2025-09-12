import re
from datetime import datetime

northern_hemisphere_meteorological_seasons_datetime = {
    "Spring": (datetime(2000, 3, 1), datetime(2000, 5, 31)),  # Year is a placeholder
    "Summer": (datetime(2000, 6, 1), datetime(2000, 8, 31)),
    "Autumn": (datetime(2000, 9, 1), datetime(2000, 11, 30)),
    "Winter": (datetime(2000, 12, 1), datetime(2001, 2, 28))  # Adjust for leap years if needed
}

southern_hemisphere_meteorological_seasons = {
    "Spring": (datetime(2000, 9, 1), datetime(2000, 11, 30)),  # Year is a placeholder
    "Summer": (datetime(2000, 12, 1), datetime(2001, 2, 28)),  # Adjust for leap years if needed
    "Autumn": (datetime(2000, 3, 1), datetime(2000, 5, 31)),
    "Winter": (datetime(2000, 6, 1), datetime(2000, 8, 31))
}

# Astronomical Seasons with datetime objects - using approximate dates
astronomical_seasons_datetime = {
    "Spring": (datetime(2000, 3, 21), datetime(2000, 6, 21)),
    "Summer": (datetime(2000, 6, 21), datetime(2000, 9, 23)),
    "Autumn": (datetime(2000, 9, 23), datetime(2000, 12, 21)),
    "Winter": (datetime(2000, 12, 21), datetime(2001, 3, 21))
}

def extract_individual_date(file_path):
    """
    Extracts a date from a file path.

    Args:
        file_path (str): File path as a string.

    Returns:
        datetime: Date extracted from the file path.
    """
    # Regular expression to match dates in format YYYYMMDD
    date_pattern = re.compile(r"\d{8}")
    
    match = date_pattern.search(file_path)
    if match:
        date_str = match.group()
        # Converting the date string to a datetime object
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        return date_obj
    else:
        return None


def extract_dates(file_paths):
    """
    Extracts dates from a list of file paths.

    Args:
        file_paths (list of str): List of file paths as strings.

    Returns:
        list of datetime: List of dates extracted from the file paths.
    """
    # Regular expression to match dates in format YYYYMMDD
    
    dates = []
    for path in file_paths:
        date = extract_individual_date(path)
        if date is not None:
            dates.append(date)
    return dates

def get_season(date, seasons_dict):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    year = date.year
    for season, (start, end) in seasons_dict.items():
        # Adjust the year for the start and end dates of the winter season
        start_year = year if season != "Winter" or date.month > 2 else year - 1
        end_year = year if season != "Winter" or date.month > 2 else year + 1
        start_date = datetime(start_year, start.month, start.day)
        end_date = datetime(end_year, end.month, end.day)

        if start_date <= date <= end_date:
            return season
    return "Season not found"

