import re
from pathlib import Path
from dotenv import dotenv_values


def to_single_line(text: str = None) -> str:
    no_surplus = re.compile(r"\s+").sub(" ", text)
    no_trailing = no_surplus.strip()

    return no_trailing

def environment_reader(env_file: str = None) -> dict:

    assert env_file is not None, 'No env file provided.'
    
    env_file_path = Path(env_file)
    
    return  dotenv_values(env_file_path)
