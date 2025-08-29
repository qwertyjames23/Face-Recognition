import sys
import os
from PyQt6 import QtWidgets, QtGui, QtCore

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, 'backend')
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from backend.db import FaceDB
from backend.utils import find_images, thumb_from_face, rel_to

from backend.cluster import Clusterer

class Indexer(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(int)

    def __init__(self, db, engine, folder):
        super().__init__()
        self.db = db
        self.engine = engine
        self.folder = folder
        self._stop = False

    def run(self):
        images = find_images(self.folder)
        total = len(images)
        added = 0
        for idx, img in enumerate(images, 1):
            if self._stop:
                break
            try:
                rel = rel_to(img, self.folder)
                img_id = self.db.ensure_image(rel, img)
                if self.db.has_faces(img_id):
                    pass
                else:
                    dets = self.engine.extract_faces(img)
                    for d in dets:
                        self.db.add_face(img_id, d['bbox'], d['embedding'])
                        added += 1
            except Exception:
                pass
            if (idx % 5 == 0) or (idx == total):
                self.progress.emit(idx, total, f'Indexing {idx}/{total} | faces added: {added}')

        # clustering
        try:
            embs = self.db.get_all_embeddings()
            if embs:
                cl = Clusterer()
                labels = cl.cluster(embs)
                self.db.apply_cluster_labels(labels)
        except Exception:
            pass

        self.finished.emit(added)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FaceRecognition — Qt')
        self.resize(1000, 700)

        toolbar = QtWidgets.QToolBar()
        self.addToolBar(toolbar)

        btn_scan = QtWidgets.QAction('Scan Folder', self)
        btn_people = QtWidgets.QAction('People', self)
        toolbar.addAction(btn_scan)
        toolbar.addAction(btn_people)

    self.status = QtWidgets.QStatusBar()
    self.setStatusBar(self.status)

    self.db = FaceDB(os.path.join(HERE, 'faces.db'))
    # lazy-load FaceEngine because importing insightface/onnxruntime
    # can raise a DLL import error on some systems. We'll create it
    # only when a scan is requested.
    self.engine = None

    btn_scan.triggered.connect(self.on_scan)
    btn_people.triggered.connect(self.on_people)

    def on_scan(self):
        dlg = QtWidgets.QFileDialog(self)
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        if dlg.exec():
            folder = dlg.selectedFiles()[0]
            self.status.showMessage(f'Indexing {folder}...')
            engine = self._get_engine()
            if engine is None:
                QtWidgets.QMessageBox.critical(self, 'Error', 'Face engine could not be loaded. Check onnxruntime and insightface installation.')
                return
            self._indexer = Indexer(self.db, engine, folder)
            self._indexer.progress.connect(lambda i,t,s: self.status.showMessage(s))
            self._indexer.finished.connect(self._on_index_finished)
            self._indexer.start()

    def _get_engine(self):
        """Lazy import and instantiate FaceEngine. Returns None on failure."""
        if self.engine is not None:
            return self.engine
        try:
            # import here to avoid top-level import errors
            from backend.face_engine import FaceEngine
            self.engine = FaceEngine()
            return self.engine
        except Exception as e:
            # log to status bar and return None; scanning will be disabled
            self.status.showMessage(f'Failed to load face engine: {e}')
            return None

    def run_index(self, folder):
        # kept for compatibility, but we use Indexer now
        pass

    def _on_index_finished(self, added):
        self.status.showMessage(f'Indexing complete. Faces added: {added}')
        # open people dialog automatically
        QtCore.QTimer.singleShot(100, self.on_people)

    def on_people(self):
        from qt_people import PeopleDialog
        dlg = PeopleDialog(self, self.db)
        dlg.exec()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
