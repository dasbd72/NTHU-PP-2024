import os


def build():
    # Check if Makefile exists
    if not os.path.exists("./Makefile"):
        return 1
    # Clean and make
    os.system("make clean")
    code = os.system("make hw4")
    if code != 0:
        return code
    return 0
