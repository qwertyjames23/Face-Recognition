from PyQt6 import QtWidgets, QtCore
import sys, os, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# ensure app root and backend are importable
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'backend'))
from qt_main import Indexer
from backend.db import FaceDB
from backend.face_engine import FaceEngine

app = QtWidgets.QApplication([])
db = FaceDB(os.path.join(ROOT,'faces.db'))
engine = FaceEngine()
folder = tempfile.mkdtemp()
print('Using temp folder:', folder)
idx = Indexer(db, engine, folder)
idx.progress.connect(lambda i,t,s: print('PROG',i,t,s))
idx.finished.connect(lambda n: print('DONE', n))
print('Starting indexer thread')
idx.start()
print('Waiting for indexer to finish...')
while idx.isRunning():
    app.processEvents()
print('Indexer thread finished')
