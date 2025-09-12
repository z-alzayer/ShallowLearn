"""
ShallowLearn utilities module
Contains cross-cutting utility functions for file operations, etc.
"""

from .file_discovery import (
    find_files_in_directory, 
    find_matching_files_by_date, 
    process_reef_data, 
    safe_filename
)

__all__ = [
    'find_files_in_directory', 
    'find_matching_files_by_date', 
    'process_reef_data', 
    'safe_filename'
]