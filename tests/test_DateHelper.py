import pytest
from datetime import datetime
import ShallowLearn.DateHelper as dh

# Sample file paths
@pytest.fixture
def sample_paths():
    return [
        "/path/to/S2A_MSIL1C_20220315T103021_N0400_R008_T30UXG_20220315T124617.SAFE",
        "some_other_file_20211101_data.zip",
        "no_date_in_this_one.txt",
        "/another/dir/IMG_20231225_ABC.tif"
    ]

# Expected dates
@pytest.fixture
def expected_dates():
    return [
        datetime(2022, 3, 15),
        datetime(2021, 11, 1),
        datetime(2023, 12, 25)
    ]

def test_extract_individual_date():
    path1 = "/path/to/S2A_MSIL1C_20220315T103021_N0400_R008_T30UXG_20220315T124617.SAFE"
    assert dh.extract_individual_date(path1) == datetime(2022, 3, 15)

    path2 = "no_date_in_this_one.txt"
    assert dh.extract_individual_date(path2) is None

    path3 = "prefix_20200101.ext"
    assert dh.extract_individual_date(path3) == datetime(2020, 1, 1)

def test_extract_dates(sample_paths, expected_dates):
    extracted = dh.extract_dates(sample_paths)
    assert len(extracted) == len(expected_dates)
    assert all(a == b for a, b in zip(extracted, expected_dates))

@pytest.mark.parametrize("date_str, hemisphere_seasons, expected_season", [
    # Southern Hemisphere Meteorological
    ("2023-09-15", dh.southern_hemisphere_meteorological_seasons, "Spring"),
    ("2024-03-01", dh.southern_hemisphere_meteorological_seasons, "Autumn"),
    ("2024-05-31", dh.southern_hemisphere_meteorological_seasons, "Autumn"),
    ("2024-06-10", dh.southern_hemisphere_meteorological_seasons, "Winter"),
    ("2024-08-31", dh.southern_hemisphere_meteorological_seasons, "Winter"),
    ("2024-09-01", dh.southern_hemisphere_meteorological_seasons, "Spring"),
    # Northern Hemisphere Meteorological
    ("2023-04-01", dh.northern_hemisphere_meteorological_seasons_datetime, "Spring"),
    ("2023-07-01", dh.northern_hemisphere_meteorological_seasons_datetime, "Summer"),
    ("2023-10-01", dh.northern_hemisphere_meteorological_seasons_datetime, "Autumn")])
def test_get_season(date_str, hemisphere_seasons, expected_season):
    test_date = datetime.strptime(date_str, "%Y-%m-%d")
    assert dh.get_season(test_date, hemisphere_seasons) == expected_season
    # Test with string input as well
    assert dh.get_season(date_str, hemisphere_seasons) == expected_season
