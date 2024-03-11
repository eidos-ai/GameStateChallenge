## Linting and Type Hint.


### Type hinter (MyPy)
Mypy is a static type checker for Python.

Type checkers help ensure that you’re using variables and functions in your code correctly. With mypy, add type hints (PEP 484) to your Python programs, and mypy will warn you when you use those types incorrectly.

Python is a dynamic language, so usually you’ll only see errors in your code when you attempt to run it. Mypy is a static checker, so it finds bugs in your programs without even running them!
Reference: https://mypy.readthedocs.io/en/stable/

#### Installation
Install using pip into your local environment:
`python3 -m pip install mypy`

#### Usage
In a shell with the environment activated run:
`mypy program.py`

This command makes mypy type check your program.py file and print out any errors it finds. Mypy will type check your code statically: this means that it will check for errors without ever running your code, just like a linter.
This also means that you are always free to ignore the errors mypy reports, if you so wish. You can always use the Python interpreter to run your code, even if mypy reports errors.

`MYPYPATH=./src mypy --strict ./src --explicit-package-bases`
If you run mypy with the *--strict* flag, you will basically never get a type related error at runtime without a corresponding mypy error, unless you explicitly circumvent mypy somehow.

However, this flag will probably be too aggressive if you are trying to add static types to a large, existing codebase

If you want to skip a line to be checked you could add `# type: ignore` at the end of the line. The same for a complete file adding the same line at the beginning.


### Black (linter)
Black is the uncompromising Python code formatter. By using it, you agree to cede control over minutiae of hand-formatting. In return, Black gives you speed, determinism, and freedom from pycodestyle nagging about formatting. You will save time and mental energy for more important matters.

Blackened code looks the same regardless of the project you're reading. Formatting becomes transparent after a while and you can focus on the content instead.
Reference: https://pypi.org/project/black/

#### Installation
Install using pip into your local environment:
`pip install black`

#### Usage
In a shell with the environment activated run:
`black ./src/{source_file_or_directory}`





