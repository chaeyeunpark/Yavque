#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import os

FILE_EXTENSIONS = ['.hpp', '.cpp']
SOURCE_DIRS = ['src', 'include', 'examples']
YAVQUE_ROOT_DIR = Path(__file__).parent.resolve()

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1].lower() == 'include_tests':
        SOURCE_DIRS += ['Tests']
    file_list = []
    for source_dir in SOURCE_DIRS:
        for ext in FILE_EXTENSIONS:
            file_list += [str(p) for p in Path(source_dir).rglob('*' + ext)]
    
    file_list = [{'name': v} for v in file_list]
    json.dump(file_list, sys.stdout)

