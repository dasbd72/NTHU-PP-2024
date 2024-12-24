import os


def build(program, quiet: bool = False):
    # Check if Makefile exists
    if not os.path.exists("./Makefile"):
        return 1
    # Clean and make
    if not quiet:
        os.system("make clean")
        code = os.system("make {}".format(program))
    else:
        os.system("make clean -q")
        code = os.system("make {} -q".format(program))
    if code != 0:
        return code
    return 0
