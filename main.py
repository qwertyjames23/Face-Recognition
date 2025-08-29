# main.py  — Quick Find mode (no pre-scan/index needed)
# Flow: Find Person -> pick reference -> pick folder -> scans ONLY for that person
# Requires: pillow, numpy, scikit-learn, opencv-python, insightface, onnxruntime

import os, sys, threading, time, shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# Prefer backend package if present; fall back to flat layout
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "backend")
for p in (HERE, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.face_engine import FaceEngine
    from backend.utils import find_images, ensure_dir, thumb_from_face, rel_to
    from backend.db import FaceDB
    
    # people UI is optional and loaded lazily; import below when needed
except ModuleNotFoundError:
    from face_engine import FaceEngine
    from utils import find_images, ensure_dir, thumb_from_face, rel_to
    from db import FaceDB

APP_TITLE = "FaceRecognition — Quick Find"
THUMB_SIZE = 140

class FaceRecApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x760")
        self.minsize(1000, 620)

        self.engine = None
        self.library_root = None
        self.worker = None
        self.stop_event = threading.Event()
        self.find_results = []  # dicts: abs_path, bbox, sim, dist

        self._build_ui()

    def _build_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        self.btn_find = ttk.Button(toolbar, text="Find Person", command=self.on_find_person)
        self.btn_people = ttk.Button(toolbar, text="People", command=self.on_people)
        self.btn_scan = ttk.Button(toolbar, text="Scan Folder", command=self.on_scan_folder)
        self.btn_export = ttk.Button(toolbar, text="Export Matches", command=self.on_find_export, state=tk.DISABLED)
        self.btn_cancel = ttk.Button(toolbar, text="Cancel", command=self.on_cancel)

        # pack People and Scan buttons at the leftmost position
        self.btn_people.pack(side=tk.LEFT, padx=4)
        self.btn_scan.pack(side=tk.LEFT, padx=4)
        for b in (self.btn_find, self.btn_export, self.btn_cancel):
            b.pack(side=tk.LEFT, padx=4)

        self.prog = ttk.Progressbar(self, mode="determinate")
        self.prog.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0,6))

        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.canvas = tk.Canvas(main, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(main, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.thumb_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.thumb_frame, anchor="nw")
        self.thumb_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.status = tk.StringVar(value="Ready. Click 'Find Person' → choose reference photo → choose folder to scan.")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)
        self.thumb_cache = {}

    # ------------ Actions ------------
    def on_cancel(self):
        self.stop_event.set()
        self._set_status("Cancelling…")

    def on_find_person(self):
        # 1) reference photo
        if not self.engine:
            self.engine = FaceEngine()

        ref_path = filedialog.askopenfilename(
            title="Choose reference photo (face)",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")]
        )
        if not ref_path:
            return

        try:
            dets = self.engine.extract_faces(ref_path)
        except Exception as e:
            messagebox.showerror("Find Person", f"Failed to read reference: {e}")
            return
        if not dets:
            messagebox.showinfo("Find Person", "No face found in the reference photo.")
            return

        # largest face if multiple
        def area(b):
            x,y,w,h = b
            return max(0,w)*max(0,h)
        det = max(dets, key=lambda d: area(d['bbox']))
        ref_emb = np.array(det['embedding'], dtype=np.float32)

        # 2) folder to scan
        folder = filedialog.askdirectory(title="Choose folder to scan for this person")
        if not folder:
            return
        self.library_root = folder

        images = find_images(folder)
        if not images:
            messagebox.showinfo("Find Person", "No images found in the chosen folder.")
            return

        self._set_status(f"Scanning {len(images)} images for this person…")
        self._run_worker(self._quick_find_worker, images, ref_emb, [0.35, 0.45, 0.55])

    def _quick_find_worker(self, images, ref_emb, thresholds):
        # l2-normalize reference
        r = ref_emb.astype("float32")
        n = float((r*r).sum()) ** 0.5
        if n > 1e-8:
            r = r / n

        total = len(images)
        matches = []
        self._set_progress(0, total)

        for idx, img_path in enumerate(images, 1):
            if self.stop_event.is_set():
                break

            try:
                dets = self.engine.extract_faces(img_path)
            except Exception:
                dets = []

            best_sim, best_det = None, None
            for d in dets:
                e = np.array(d["embedding"], dtype=np.float32)
                den = float((e*e).sum()) ** 0.5 or 1e-8
                sim = float((e @ r) / den)   # cosine similarity
                if (best_sim is None) or (sim > best_sim):
                    best_sim, best_det = sim, d

            if best_sim is not None:
                dist = 1.0 - best_sim
                # progressive thresholds (lower=better)
                if any(dist <= t for t in thresholds):
                    matches.append({"abs_path": img_path, "bbox": best_det["bbox"], "sim": best_sim, "dist": dist})

            if (idx % 10 == 0) or (idx == total):
                self._set_progress(idx, total, f"Scanning {idx}/{total} | matches: {len(matches)}")

        matches.sort(key=lambda x: -x["sim"])

        def done():
            self.find_results = matches
            if self.stop_event.is_set():
                self._set_status(f"Scan cancelled at {idx}/{total}. Matches so far: {len(matches)}")
                self.stop_event.clear()
            else:
                if not matches:
                    self._set_status("No matches found for this person. Try a clearer reference photo.")
                    for w in self.thumb_frame.winfo_children():
                        w.destroy()
                else:
                    self._set_status(f"Found {len(matches)} possible matches (showing top {min(120, len(matches))}).")
                    self._show_find_results(matches[:120])
                    self.btn_export.configure(state=tk.NORMAL)
        self.after(0, done)

    def on_find_export(self):
        if not self.find_results:
            messagebox.showinfo("Export Matches", "No matches to export.")
            return
        out_root = filedialog.askdirectory(
            title="Choose export folder",
            initialdir=os.path.join(self.library_root or "", "exports")
        )
        if not out_root:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(out_root, f"FindPerson_{stamp}")
        os.makedirs(out_dir, exist_ok=True)
        seen, copied = set(), 0
        for rec in self.find_results:
            src = rec["abs_path"]
            if src in seen or not os.path.exists(src):
                continue
            seen.add(src)
            dst = os.path.join(out_dir, os.path.basename(src))
            base, ext = os.path.splitext(dst)
            j = 1
            while os.path.exists(dst):
                dst = f"{base}_{j}{ext}"; j += 1
            try:
                shutil.copy2(src, dst); copied += 1
            except Exception:
                pass
        messagebox.showinfo("Export Matches", f"Exported {copied} files to {out_dir}")
        try:
            if os.name == "nt":
                os.startfile(out_dir)
        except Exception:
            pass

    def on_people(self):
        # lazy-create DB instance
        if not hasattr(self, 'db') or self.db is None:
            dbpath = os.path.join(HERE, 'faces.db')
            self.db = FaceDB(dbpath)
        try:
            from people_window import PeopleWindow
        except Exception:
            # try backend folder
            from backend.people_window import PeopleWindow
        PeopleWindow(self, self.db)

    def on_scan_folder(self):
        folder = filedialog.askdirectory(title='Choose folder to index into DB')
        if not folder:
            return
        # ensure DB
        if not hasattr(self, 'db') or self.db is None:
            dbpath = os.path.join(HERE, 'faces.db')
            self.db = FaceDB(dbpath)
        # create engine if needed
        if not self.engine:
            self.engine = FaceEngine()
        # run indexing in background
        self._run_worker(self._index_folder_worker, folder)

    def _open_people_window(self):
        """Robustly import and open the PeopleWindow UI on the main thread."""
        try:
            # try app-local module first
            from people_window import PeopleWindow
        except Exception:
            try:
                # try backend package
                from backend.people_window import PeopleWindow
            except Exception:
                # final fallback: load by filepath
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location('people_window', os.path.join(HERE, 'people_window.py'))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    PeopleWindow = getattr(mod, 'PeopleWindow')
                except Exception:
                    return
        try:
            PeopleWindow(self, self.db)
        except Exception:
            # if creating the window fails, silently ignore to avoid crashing the app
            pass

    def _index_folder_worker(self, folder):
        images = find_images(folder)
        total = len(images)
        self._set_progress(0, total)
        added = 0
        for idx, img in enumerate(images, 1):
            try:
                rel = rel_to(img, folder)
                img_id = self.db.ensure_image(rel, img)
                if self.db.has_faces(img_id):
                    # already processed
                    pass
                else:
                    dets = self.engine.extract_faces(img)
                    for d in dets:
                        self.db.add_face(img_id, d['bbox'], d['embedding'])
                        added += 1
            except Exception:
                pass
            if (idx % 10 == 0) or (idx == total):
                self._set_progress(idx, total, f'Indexing {idx}/{total} | faces added: {added}')
        self._set_status(f'Indexing complete. Faces added: {added}')
        # --- run clustering on all embeddings and apply labels ---
        try:
            embs = self.db.get_all_embeddings()
            if embs:
                from backend.cluster import Clusterer
                cl = Clusterer()
                labels = cl.cluster(embs)
                self.db.apply_cluster_labels(labels)
                clusters = self.db.list_clusters()
                cnt = len(clusters)
                self._set_status(f'Indexing complete. Faces added: {added}. Found {cnt} people.')
            else:
                self._set_status(f'Indexing complete. Faces added: {added}. No faces found.')
        except Exception:
            # don't crash worker on clustering errors
            pass

        # always try to open People window (even if clustering errored)
        self.after(0, self._open_people_window)

    # ------------ UI helpers ------------
    def _run_worker(self, target, *args):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "A job is already running.")
            return
        self.worker = threading.Thread(target=target, args=args, daemon=True)
        self.worker.start()

    def _set_status(self, text):
        if threading.current_thread() is threading.main_thread():
            self.status.set(text)
        else:
            self.after(0, lambda: self.status.set(text))

    def _set_progress(self, val, total, status=None):
        def upd():
            self.prog.configure(maximum=total, value=val)
            if status:
                self.status.set(status)
        if threading.current_thread() is threading.main_thread():
            upd()
        else:
            self.after(0, upd)

    def _show_find_results(self, results):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self.thumb_cache.clear()

        self.thumb_frame.update_idletasks()
        width = self.canvas.winfo_width() or 900
        cols = max(1, width // (THUMB_SIZE + 16))
        r = c = 0
        for rec in results:
            path, bbox = rec["abs_path"], rec["bbox"]
            try:
                img = Image.open(path).convert("RGB")
                thumb = thumb_from_face(img, bbox, THUMB_SIZE)
                tkimg = ImageTk.PhotoImage(thumb)
            except Exception:
                continue

            frame = ttk.Frame(self.thumb_frame, relief="flat", cursor="hand2")
            lbl = ttk.Label(frame, image=tkimg, cursor="hand2")
            lbl.image = tkimg
            lbl.pack(side=tk.TOP, padx=2, pady=2)
            cap = ttk.Label(frame, text=os.path.basename(path), cursor="hand2")
            cap.pack(side=tk.TOP)

            frame.grid(row=r, column=c, padx=6, pady=6)
            self.thumb_cache[(r,c)] = tkimg

            for w in (frame, lbl, cap):
                w.bind("<Double-Button-1>", lambda e, p=path: self._open_folder(p))

            c += 1
            if c >= cols:
                c = 0
                r += 1

    def _open_folder(self, path):
        try:
            if os.name == "nt":
                os.startfile(os.path.dirname(path))
        except Exception:
            pass

if __name__ == "__main__":
    app = FaceRecApp()
    app.mainloop()
