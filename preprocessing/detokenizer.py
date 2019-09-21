import sys


if __name__ == "__main__":
    for line in sys.stdin:
        if line.strip() != "":
            if '▁▁' in line:
                line = line.strip().replace(' ', '').replace('▁▁', ' ').replace('_', '').strip()
            else:
                line = line.strip().replace(' ', '').replace('_', ' ').strip()

            sys.stdout.write(line + '\n')
        else:
            sys.stdout.write('\n')