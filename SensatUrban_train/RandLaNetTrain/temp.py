 
import sys
from pathlib import Path
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CODE_PATH = str(Path(__file__).absolute().parent.parent)
print(CODE_PATH)
print("Python Import Paths:")
for path in sys.path:
    print(path)

from SensatUrban.helper_ply import read_ply
