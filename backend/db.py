import os
import sqlite3
import json
import shutil
import numpy as np
import threading

def adapt_array(arr):
    return arr.tobytes()

def convert_array(blob):
    return np.frombuffer(blob, dtype=np.float32)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

class FaceDB:
    def __init__(self, path: str):
        self.path = path
        # Allow use from worker thread
        self.conn = sqlite3.connect(
            self.path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False
        )
        self.lock = threading.Lock()
        self._migrate()

    def _migrate(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS images(
                id INTEGER PRIMARY KEY,
                rel_path TEXT UNIQUE,
                abs_path TEXT,
                mtime REAL
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS faces(
                id INTEGER PRIMARY KEY,
                image_id INTEGER,
                bbox TEXT,
                embedding ARRAY,
                cluster_id INTEGER DEFAULT NULL,
                FOREIGN KEY(image_id) REFERENCES images(id)
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS clusters(
                id INTEGER PRIMARY KEY,
                label TEXT
            )""")
            self.conn.commit()

    # ---------- Images ----------
    def ensure_image(self, rel_path: str, abs_path: str) -> int:
        st = os.stat(abs_path)
        mtime = st.st_mtime
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO images(rel_path, abs_path, mtime) VALUES(?,?,?)",
                (rel_path, abs_path, mtime),
            )
            cur.execute("SELECT id FROM images WHERE rel_path=?", (rel_path,))
            row = cur.fetchone()
            self.conn.commit()
        return row[0]

    def has_faces(self, image_id: int) -> bool:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(1) FROM faces WHERE image_id=?", (image_id,))
            got = cur.fetchone()[0] > 0
        return got

    # ---------- Faces ----------
    def add_face(self, image_id: int, bbox, embedding: np.ndarray):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO faces(image_id, bbox, embedding) VALUES(?,?,?)",
                (image_id, json.dumps(bbox), embedding),
            )
            self.conn.commit()

    def get_all_embeddings(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id, embedding FROM faces")
            rows = cur.fetchall()
        self._face_ids = [r[0] for r in rows]
        return [r[1] for r in rows]

    def apply_cluster_labels(self, labels):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM clusters")

            from collections import defaultdict
            groups = defaultdict(list)
            for fid, lab in zip(self._face_ids, labels):
                if lab < 0:
                    continue
                groups[int(lab)].append(fid)

            next_id = 1
            for _, members in groups.items():
                cur.execute(
                    "INSERT INTO clusters(id, label) VALUES(?,?)",
                    (next_id, f"Person #{next_id}"),
                )
                for fid in members:
                    cur.execute("UPDATE faces SET cluster_id=? WHERE id=?", (next_id, fid))
                next_id += 1

            for fid, lab in zip(self._face_ids, labels):
                if lab < 0:
                    cur.execute("UPDATE faces SET cluster_id=NULL WHERE id=?", (fid,))

            self.conn.commit()

    # ---------- Clusters ----------
    def list_clusters(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT c.id, c.label, COUNT(f.id) as cnt
                FROM clusters c
                LEFT JOIN faces f ON f.cluster_id = c.id
                GROUP BY c.id, c.label
                ORDER BY cnt DESC
            """)
            rows = cur.fetchall()
        return rows

    def rename_cluster(self, cluster_id: int, new_label: str):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("UPDATE clusters SET label=? WHERE id=?", (new_label, cluster_id))
            self.conn.commit()

    def merge_clusters(self, keep_id: int, merged_ids):
        with self.lock:
            cur = self.conn.cursor()
            for mid in merged_ids:
                cur.execute("UPDATE faces SET cluster_id=? WHERE cluster_id=?", (keep_id, mid))
                cur.execute("DELETE FROM clusters WHERE id=?", (mid,))
            self.conn.commit()

    def get_faces_by_cluster(self, cluster_id: int):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT f.bbox, i.abs_path
                FROM faces f
                JOIN images i ON i.id = f.image_id
                WHERE f.cluster_id=?
            """, (cluster_id,))
            rows = cur.fetchall()
        return [{"bbox": json.loads(b), "abs_path": p} for (b, p) in rows]

    def get_recent_faces(self, limit: int = 50):
        """Return the most recently added faces (by face id) as a list of dicts
        containing 'bbox' and 'abs_path'. This is useful to preview faces
        immediately after an indexing run.
        """
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT f.bbox, i.abs_path
                FROM faces f
                JOIN images i ON i.id = f.image_id
                ORDER BY f.id DESC
                LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
        return [{"bbox": json.loads(b), "abs_path": p} for (b, p) in rows]

    # ---------- Suggestions ----------
    def _cluster_centroid(self, cluster_id: int):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT embedding FROM faces WHERE cluster_id=?", (cluster_id,))
            embs = [r[0] for r in cur.fetchall()]
        if not embs:
            return None
        X = np.vstack(embs).astype(np.float32)
        x = X.mean(axis=0)
        n = np.linalg.norm(x) + 1e-9
        return x / n

    def suggest_merges(self, thresh=0.35, topk=50):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT id FROM clusters")
            ids = [r[0] for r in cur.fetchall()]
        cents = [self._cluster_centroid(cid) for cid in ids]
        out = []
        import itertools
        for a, b in itertools.combinations(range(len(ids)), 2):
            ca, cb = cents[a], cents[b]
            if ca is None or cb is None:
                continue
            sim = float((ca * cb).sum())
            if (1.0 - sim) <= thresh:
                out.append((ids[a], ids[b], sim))
        out.sort(key=lambda x: -x[2])
        return out[:topk]

    # ---------- Export ----------
    def export_cluster(self, cluster_id: int, library_root: str, out_root: str) -> int:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT label FROM clusters WHERE id=?", (cluster_id,))
            row = cur.fetchone()
        label = row[0] if row else f"Person_{cluster_id}"
        person_dir = os.path.join(out_root, label.replace("/", "_"))
        os.makedirs(person_dir, exist_ok=True)

        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT DISTINCT i.abs_path
                FROM faces f
                JOIN images i ON i.id = f.image_id
                WHERE f.cluster_id=?
            """, (cluster_id,))
            rows = cur.fetchall()

        count = 0
        for (src,) in rows:
            if not os.path.exists(src):
                continue
            dst = os.path.join(person_dir, os.path.basename(src))
            base, ext = os.path.splitext(dst)
            j = 1
            while os.path.exists(dst):
                dst = f"{base}_{j}{ext}"
                j += 1
            try:
                shutil.copy2(src, dst)
                count += 1
            except Exception:
                pass
        return count