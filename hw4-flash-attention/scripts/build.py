import os


def build(quiet: bool = False):
    # Check if Makefile exists
    if not os.path.exists("./Makefile"):
        return 1
    # Clean and make
    if not quiet:
        os.system("make clean")
        code = os.system("make hw4")
    else:
        os.system("make clean > /dev/null 2>&1")
        code = os.system("make hw4 > /dev/null 2>&1")
    if code != 0:
        return code
    return 0
