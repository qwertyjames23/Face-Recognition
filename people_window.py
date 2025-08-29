import os
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
from PIL import Image, ImageTk
from backend.utils import thumb_from_face

class PeopleWindow(tk.Toplevel):
    def __init__(self, parent, db):
        super().__init__(parent)
        self.title('People')
        self.geometry('900x600')
        self.db = db

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        self.btn_refresh = ttk.Button(top, text='Refresh', command=self.refresh)
        self.btn_rename = ttk.Button(top, text='Rename', command=self.rename)
        self.btn_merge = ttk.Button(top, text='Merge', command=self.merge)
        self.btn_export = ttk.Button(top, text='Export', command=self.export)
        for b in (self.btn_refresh, self.btn_rename, self.btn_merge, self.btn_export):
            b.pack(side=tk.LEFT, padx=4)

        self.tree = ttk.Treeview(self, columns=('label', 'count'), show='headings', selectmode='extended')
        self.tree.heading('label', text='Label')
        self.tree.heading('count', text='Photos')
        self.tree.column('label', width=300)
        self.tree.column('count', width=80, anchor='center')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=6)

        self.preview = ttk.Frame(self)
        self.preview.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=6, pady=6)

        self.canvas = tk.Canvas(self.preview, width=320)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bind selection and double-click handlers
        self.tree.bind('<<TreeviewSelect>>', self.on_select)
        self.tree.bind('<Double-1>', self.on_tree_double)

        self._thumbs = []
        self.refresh()

    def refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        rows = self.db.list_clusters()
        for cid, label, cnt in rows:
            self.tree.insert('', 'end', iid=str(cid), values=(label, cnt))

    def _get_selected_cluster_ids(self):
        return [int(i) for i in self.tree.selection()]

    def rename(self):
        sel = self._get_selected_cluster_ids()
        if len(sel) != 1:
            messagebox.showinfo('Rename', 'Select exactly one person to rename.')
            return
        cid = sel[0]
        cur = self.db.conn.cursor()
        cur.execute('SELECT label FROM clusters WHERE id=?', (cid,))
        row = cur.fetchone()
        cur_label = row[0] if row else ''
        new = simpledialog.askstring('Rename', 'New name for person:', initialvalue=cur_label, parent=self)
        if new:
            self.db.rename_cluster(cid, new)
            self.refresh()

    def merge(self):
        sels = self._get_selected_cluster_ids()
        if len(sels) < 2:
            messagebox.showinfo('Merge', 'Select two or more people to merge into the first selected.')
            return
        keep = sels[0]
        merged = [s for s in sels[1:]]
        self.db.merge_clusters(keep, merged)
        self.refresh()

    def export(self):
        sel = self._get_selected_cluster_ids()
        if len(sel) != 1:
            messagebox.showinfo('Export', 'Select a single person to export.')
            return
        cid = sel[0]
        out = filedialog.askdirectory(title='Choose export folder')
        if not out:
            return
        libroot = os.getcwd()
        n = self.db.export_cluster(cid, libroot, out)
        messagebox.showinfo('Export', f'Exported {n} files to {out}')

    def on_select(self, _ev=None):
        sels = self._get_selected_cluster_ids()
        if not sels:
            return
        cid = sels[0]
        faces = self.db.get_faces_by_cluster(cid)
        # show up to 16 thumbs
        for w in self.preview.winfo_children():
            w.destroy()
        self._thumbs.clear()
        cols = 4
        r = c = 0
        for rec in faces[:16]:
            path = rec['abs_path']
            bbox = rec['bbox']
            try:
                img = Image.open(path).convert('RGB')
                thumb = thumb_from_face(img, bbox, size=140)
                tkimg = ImageTk.PhotoImage(thumb)
            except Exception:
                continue
            lbl = ttk.Label(self.preview, image=tkimg)
            lbl.image = tkimg
            lbl.grid(row=r, column=c, padx=4, pady=4)
            c += 1
            if c >= cols:
                c = 0; r += 1

    def on_tree_double(self, ev=None):
        iid = self.tree.focus()
        if not iid:
            return
        try:
            cid = int(iid)
        except Exception:
            return
        PersonDetailWindow(self, self.db, cid)


class PersonDetailWindow(tk.Toplevel):
    def __init__(self, parent, db, cluster_id):
        super().__init__(parent)
        self.title(f'Person {cluster_id}')
        self.geometry('1000x700')
        self.db = db
        self.cluster_id = cluster_id

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Button(top, text='Export', command=self.export).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self)
        self.scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0,0), window=self.frame, anchor='nw')
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.frame.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))

        self._thumbs = []
        self._populate()

    def _populate(self):
        for w in self.frame.winfo_children():
            w.destroy()
        faces = self.db.get_faces_by_cluster(self.cluster_id)
        cols = 5
        r = c = 0
        for rec in faces:
            path = rec['abs_path']
            bbox = rec['bbox']
            try:
                img = Image.open(path).convert('RGB')
                thumb = thumb_from_face(img, bbox, size=160)
                tkimg = ImageTk.PhotoImage(thumb)
            except Exception:
                continue
            lbl = ttk.Label(self.frame, image=tkimg, cursor='hand2')
            lbl.image = tkimg
            lbl.grid(row=r, column=c, padx=6, pady=6)
            lbl.bind('<Double-1>', lambda e, p=path: self._open(p))
            cap = ttk.Label(self.frame, text=os.path.basename(path))
            cap.grid(row=r+1, column=c, padx=6, pady=(0,8))
            c += 1
            if c >= cols:
                c = 0; r += 2

    def _open(self, path):
        try:
            if os.name == 'nt':
                os.startfile(path)
        except Exception:
            pass

    def export(self):
        out = filedialog.askdirectory(title='Choose export folder')
        if not out:
            return
        n = self.db.export_cluster(self.cluster_id, os.getcwd(), out)
        messagebox.showinfo('Export', f'Exported {n} files to {out}', parent=self)
