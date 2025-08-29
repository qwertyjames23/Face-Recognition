import sys
import os
from PyQt6 import QtWidgets, QtGui, QtCore
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, 'backend')
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from backend.db import FaceDB
from backend.utils import find_images, thumb_from_face, rel_to

from backend.cluster import Clusterer


class ThumbnailResultsDialog(QtWidgets.QDialog):
    """Simple scrollable grid dialog that shows thumbnails for search results.

    results: list of (similarity, abs_path, bbox)
    Clicking a thumbnail opens the image with the OS default viewer.
    """
    def __init__(self, parent, results, thumb_size=120, cols=6):
        super().__init__(parent)
        self.setWindowTitle('Find Person - Matches')
        self.resize(900, 600)

        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(f'Showing top {len(results)} matches')
        layout.addWidget(info)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        # tighten spacing so thumbnails are close together
        grid.setSpacing(6)
        grid.setContentsMargins(6, 6, 6, 6)

        r = c = 0
        for sim, path, bbox in results:
            try:
                import cv2
                arr = cv2.imread(path)
                if arr is None:
                    continue
                # face_engine uses cv2 coordinates (x1,y1,x2,y2)
                x1, y1, x2, y2 = bbox
                h_img, w_img = arr.shape[:2]
                x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(w_img, int(x2)); y2 = min(h_img, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                face = arr[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                fh, fw = face_rgb.shape[:2]

                # convert to QImage
                bytes_per_line = fw * 3
                qimg = QtGui.QImage(face_rgb.data, fw, fh, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()
                pix = QtGui.QPixmap.fromImage(qimg)

                # scale while keeping aspect ratio, then center on a square canvas
                scaled = pix.scaled(thumb_size, thumb_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                canvas = QtGui.QPixmap(thumb_size, thumb_size)
                canvas.fill(QtGui.QColor('#f5f5f5'))
                painter = QtGui.QPainter(canvas)
                xoff = (thumb_size - scaled.width()) // 2
                yoff = (thumb_size - scaled.height()) // 2
                painter.drawPixmap(xoff, yoff, scaled)
                painter.end()

                btn = QtWidgets.QPushButton()
                # remove visible button chrome and focus border
                btn.setFlat(True)
                btn.setStyleSheet('border: none; padding: 0; margin: 0;')
                btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
                btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
                btn.setFixedSize(thumb_size, thumb_size)
                icon = QtGui.QIcon(canvas)
                btn.setIcon(icon)
                btn.setIconSize(QtCore.QSize(thumb_size, thumb_size))
                btn.setToolTip(f"{os.path.basename(path)}")
                btn._path = path
                btn.clicked.connect(lambda _checked, p=path: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(p)))

                v = QtWidgets.QVBoxLayout()
                # zero margins so items sit tightly together
                v.setContentsMargins(0, 2, 0, 2)
                v.setSpacing(2)
                v.addWidget(btn)
                fname = os.path.basename(path)
                lbl = QtWidgets.QLabel(fname)
                lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                v.addWidget(lbl)

                container = QtWidgets.QWidget()
                container.setLayout(v)
                container.setContentsMargins(0, 0, 0, 0)
                container.setStyleSheet('background: transparent;')
                grid.addWidget(container, r, c)
                c += 1
                if c >= cols:
                    c = 0
                    r += 1
            except Exception:
                continue

        scroll.setWidget(w)
        layout.addWidget(scroll)
        btns = QtWidgets.QHBoxLayout()
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(close)
        layout.addLayout(btns)


class Indexer(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(int)

    def __init__(self, db, engine, folder):
        super().__init__()
        self.db = db
        self.engine = engine
        self.folder = folder
        self._stop = False

    def stop(self):
        self._stop = True

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

        # create actions
        btn_find = QtGui.QAction('Find Person', self)
        btn_scan = QtGui.QAction('Scan Folder', self)
        btn_people = QtGui.QAction('People', self)
        btn_export = QtGui.QAction('Export Matches', self)
        btn_cancel = QtGui.QAction('Cancel', self)
        toolbar.addAction(btn_find)
        toolbar.addAction(btn_scan)
        toolbar.addAction(btn_people)
        toolbar.addAction(btn_export)
        toolbar.addAction(btn_cancel)

        # status bar and DB should be initialized as part of the window
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        self.db = FaceDB(os.path.join(HERE, 'faces.db'))
        # lazy-load FaceEngine because importing insightface/onnxruntime
        # can raise a DLL import error on some systems. We'll create it
        # only when a scan is requested.
        self.engine = None

        # wire actions
        btn_scan.triggered.connect(self.on_scan)
        btn_people.triggered.connect(self.on_people)
        btn_find.triggered.connect(self.on_find_person)
        btn_export.triggered.connect(self.on_people)
        btn_cancel.triggered.connect(self.on_cancel)

        # indexer placeholder
        self._indexer = None

    def on_cancel(self):
        # Stop any running indexer thread
        if self._indexer is not None:
            try:
                self._indexer.stop()
                self.status.showMessage('Cancelling...')
            except Exception:
                pass

    def on_find_person(self):
        # Quick find: pick reference photo and folder, then scan folder for similar faces
        engine = self._get_engine()
        if engine is None:
            QtWidgets.QMessageBox.critical(self, 'Error', 'Face engine could not be loaded. Check onnxruntime and insightface installation.')
            return
        ref_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose reference photo', filter='Images (*.jpg *.jpeg *.png *.bmp *.webp)')
        if not ref_path:
            return
        try:
            dets = engine.extract_faces(ref_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to read reference: {e}')
            return
        if not dets:
            QtWidgets.QMessageBox.information(self, 'Find Person', 'No face found in the reference photo.')
            return
        # use largest face
        def area(b):
            x,y,w,h = b
            return max(0,w)*max(0,h)
        det = max(dets, key=lambda d: area(d['bbox']))
        ref_emb = np.array(det['embedding'], dtype=np.float32)

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose folder to scan for this person')
        if not folder:
            return

        imgs = find_images(folder)
        results = []
        for img in imgs:
            try:
                ds = engine.extract_faces(img)
                for d in ds:
                    emb = np.array(d['embedding'], dtype=np.float32)
                    # cosine similarity
                    sim = float((ref_emb * emb).sum() / (np.linalg.norm(ref_emb) * np.linalg.norm(emb) + 1e-9))
                    results.append((sim, img, d['bbox']))
            except Exception:
                pass
        results.sort(key=lambda x: -x[0])
        # filter by similarity threshold to avoid returning every image
        threshold = 0.50  # show matches with cosine similarity >= threshold
        filtered = [r for r in results if r[0] >= threshold]
        top = filtered[:50]
        if not top:
            QtWidgets.QMessageBox.information(self, 'Find Person', 'No matches found above similarity threshold (0.50). Try a different reference photo or lower the threshold in settings.')
            return
        dlg = ThumbnailResultsDialog(self, top)
        dlg.exec()

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
