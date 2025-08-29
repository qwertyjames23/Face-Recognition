import sys, traceback, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import qt_main
    print('Imported qt_main successfully')
except Exception:
    traceback.print_exc()
    raise
