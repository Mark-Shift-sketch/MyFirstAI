import subprocess
import sys
import os

def main():
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print('Starting retrain (this runs train.py)...')
    subprocess.run([sys.executable, os.path.join(cwd, 'train.py')], check=False)

if __name__ == '__main__':
    main()
