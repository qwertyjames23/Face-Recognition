# Face Recognition (PyQt6)

A simple, offline photo organizer that groups images by person—similar to Google Photos’ “People” view—built with Python + PyQt6. It detects faces, lets you rename people, merge duplicates, and keeps a fast local index/database for quick search.

Status: WIP (open source) • License: MIT

✨ Features

- 🧑‍🤝‍🧑 People view – auto-group photos by detected faces
- ✏️ Rename / Merge – label a person and merge duplicate clusters
- 🗂️ Local indexing – stores results in a lightweight DB (faces.db)
- 🖥️ Desktop GUI – PyQt6 interface (no internet required)
- 📁 Folder-based – point to a folder of images; it will scan/index
- 🔍 Search – filter by person/name once labeled

📦 Tech Stack

- Python 3.9+
- PyQt6 – GUI
- OpenCV / face engine – detection & embeddings (see backend/)
- SQLite – local metadata DB (faces.db)

Note: You can swap in any face model/engine under backend/face_engine.py.

## 💖 Support

If you find this project useful, consider supporting its development:

- 💸 [PayPal](https://paypal.me/RaffyjamesAdams)

Your support helps me maintain and improve this project. 🙏

### 🇵🇭 GCash (QR)

You can also support the project via GCash by scanning the QR below.

Place your QR image file at `assets/gcash-qr.png` and it will be displayed here:

![GCash QR](assets/gcash-qr.png)

If you prefer, include a payment note or phone number in the image filename or the repository README.
