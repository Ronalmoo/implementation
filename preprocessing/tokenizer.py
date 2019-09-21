import sys, fileinput
from konlpy.tag import Okt

t = Okt()


if __name__ == "__main__":
    for line in fileinput.input('input.en.txt'):
        if line.strip() != "":
            tokens = t.morphs(line.strip())

            sys.stdout.write(" ".join(tokens) + "\n")
        else:
            sys.stdout.write('\n')

