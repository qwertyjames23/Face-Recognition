from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6 import QtWidgets, QtGui, QtCore
import os

from backend.utils import thumb_from_face


class PeopleDialog(QtWidgets.QDialog):
    def __init__(self, parent, db):
        super().__init__(parent)
        self.setWindowTitle('People')
        self.resize(800, 600)
        self.db = db

        layout = QtWidgets.QVBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        layout.addWidget(self.list)

        btns = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton('Refresh')
        self.btn_view = QtWidgets.QPushButton('View')
        self.btn_rename = QtWidgets.QPushButton('Rename')
        self.btn_merge = QtWidgets.QPushButton('Merge')
        self.btn_export = QtWidgets.QPushButton('Export')
        btns.addWidget(self.btn_refresh)
        btns.addWidget(self.btn_view)
        btns.addWidget(self.btn_rename)
        btns.addWidget(self.btn_merge)
        btns.addWidget(self.btn_export)
        layout.addLayout(btns)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_view.clicked.connect(self.view_selected)
        self.btn_rename.clicked.connect(self.rename_selected)
        self.btn_merge.clicked.connect(self.merge_selected)
        self.btn_export.clicked.connect(self.export_selected)

        self.refresh()

    def refresh(self):
        self.list.clear()
        rows = self.db.list_clusters()
        if not rows:
            # show a non-selectable placeholder so the dialog isn't filled with stale entries
            item = QtWidgets.QListWidgetItem('(No people indexed yet — scan a folder)')
            # remove selection/interaction flags
            item.setFlags(QtCore.Qt.ItemFlag(0))
            self.list.addItem(item)
            # disable action buttons until there are real entries
            self.btn_view.setEnabled(False)
            self.btn_rename.setEnabled(False)
            self.btn_merge.setEnabled(False)
            self.btn_export.setEnabled(False)
        else:
            for cid, label, cnt in rows:
                self.list.addItem(f'{cid}: {label} ({cnt})')
            # enable action buttons
            self.btn_view.setEnabled(True)
            self.btn_rename.setEnabled(True)
            self.btn_merge.setEnabled(True)
            self.btn_export.setEnabled(True)

    def _get_selected_ids(self):
        out = []
        for it in self.list.selectedItems():
            out.append(int(it.text().split(':', 1)[0]))
        return out

    def rename_selected(self):
        ids = self._get_selected_ids()
        if len(ids) != 1:
            QtWidgets.QMessageBox.information(self, 'Rename', 'Select exactly one person to rename.')
            return
        cid = ids[0]
        cur = self.db.conn.cursor()
        cur.execute('SELECT label FROM clusters WHERE id=?', (cid,))
        row = cur.fetchone()
        cur_label = row[0] if row else ''
        text, ok = QtWidgets.QInputDialog.getText(self, 'Rename', 'New name:', text=cur_label)
        if ok and text:
            self.db.rename_cluster(cid, text)
            self.refresh()

    def merge_selected(self):
        ids = self._get_selected_ids()
        if len(ids) < 2:
            QtWidgets.QMessageBox.information(self, 'Merge', 'Select two or more people to merge into the first selected.')
            return
        keep = ids[0]
        merged = ids[1:]
        self.db.merge_clusters(keep, merged)
        self.refresh()

    def export_selected(self):
        ids = self._get_selected_ids()
        if len(ids) != 1:
            QtWidgets.QMessageBox.information(self, 'Export', 'Select a single person to export.')
            return
        cid = ids[0]
        out = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose export folder')
        if not out:
            return
        n = self.db.export_cluster(cid, os.getcwd(), out)
        QtWidgets.QMessageBox.information(self, 'Export', f'Exported {n} files to {out}')

    def view_selected(self):
        it = self.list.currentItem()
        if not it:
            return
        cid = int(it.text().split(':', 1)[0])
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f'Person {cid}')
        lay = QtWidgets.QGridLayout(dlg)
        faces = self.db.get_faces_by_cluster(cid)
        r = c = 0
        for rec in faces[:50]:
            path = rec['abs_path']
            try:
                from PIL import Image
                img = Image.open(path).convert('RGB')
                thumb = thumb_from_face(img, rec['bbox'], size=120)
                data = thumb.tobytes('raw', 'RGB')
                qimg = QtGui.QImage(data, thumb.width, thumb.height, QtGui.QImage.Format.Format_RGB888)
                pix = QtGui.QPixmap.fromImage(qimg)
                lbl = QtWidgets.QLabel()
                lbl.setPixmap(pix)
                lay.addWidget(lbl, r, c)
                c += 1
                if c >= 6:
                    c = 0
                    r += 1
            except Exception:
                pass
        dlg.exec()
