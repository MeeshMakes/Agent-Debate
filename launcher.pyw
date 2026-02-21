# launcher.pyw — runs with pythonw.exe (no console window)
import sys, os

# Ensure the project root is on the path
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

os.chdir(_root)

from app.main import main
main()
