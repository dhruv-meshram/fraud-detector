import os
import re

patterns = [
    # Refactor prj references
    (re.compile(r'\bfrom prj\b'), 'from fraud_detector'),
    (re.compile(r'\bimport prj\b'), 'import fraud_detector'),
    # Refactor algorithms references
    (re.compile(r'\bfrom algorithms\b'), 'from fraud_detector.algorithms'),
    (re.compile(r'\bimport algorithms\b'), 'import fraud_detector.algorithms'),
    # Refactor ml references
    (re.compile(r'\bfrom ml\b'), 'from fraud_detector.ml'),
    (re.compile(r'\bimport ml\b'), 'import fraud_detector.ml'),
]

def refactor_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    for pattern, replacement in patterns:
        new_content = pattern.sub(replacement, new_content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Refactored: {filepath}")

def main():
    exclude_dirs = {'.git', 'venv', '.pytest_cache', 'prj.egg-info'}
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py') or file.endswith('.sh') or file.endswith('.toml'):
                # Avoid refactoring this script itself
                if file == 'refactor_imports.py':
                    continue
                refactor_file(os.path.join(root, file))

if __name__ == '__main__':
    main()
