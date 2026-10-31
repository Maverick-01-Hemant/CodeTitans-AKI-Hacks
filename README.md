# CodeTitans · Smart Face-Recognition Attendance

Consent-based attendance system built with **Flask + face_recognition (dlib) + SQLite**.
Runs 100% on your own computer: no cloud, no third parties, **no photos stored**.

## Features

| Feature | What it does |
|---|---|
| Consent-based registration | A person can't be registered unless they tick the consent box; consent time is stored |
| Duplicate-face protection | The same face can't be registered under a second ID (stops proxy registration) |
| Live recognition | Multi-face detection; a face must match in consecutive frames before it counts |
| **Liveness check (blink)** | Optional: person must blink once, so a printed photo or phone screen doesn't work |
| Late marking | Set "Late after HH:MM"; later marks are flagged Late |
| Manual mark / undo | Fallback for people who opt out of face scanning, or for corrections |
| Dashboard | Registered, present, late/absent, attendance % today, 7-day chart, per-person % |
| CSV export | One day or all days (formula-injection safe) |
| Right to erasure | Delete one person (face data + attendance) or wipe everything |

## Install (Windows / macOS / Linux)

1. Install **Python 3.10 – 3.12** (tick "Add Python to PATH" on Windows).
2. In this folder:

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install flask numpy pillow dlib-bin face_recognition_models "setuptools<81"
pip install face_recognition --no-deps
```

`dlib-bin` is a prebuilt dlib, so no CMake or C++ compiler is needed.

## Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** and allow camera access.

## How to use

1. **Register**: ID, name, consent box, *Capture & register* (5 snapshots).
2. **Take Attendance**: *Start scanning*. With the liveness check on, blink once when the box turns amber.
3. Optionally set **Late after** (e.g. 09:15).
4. **Records**: pick a date, search, mark/undo manually, export CSV.

## How it works

```
Browser camera ──snapshot (400px JPEG)──▶ Flask /api/recognize
                                            │ face_recognition: locate face → 128-d encoding
                                            │ distance to stored encodings ≤ 0.5 → candidate
                                            │ 2 consecutive matches (+ blink if liveness on)
                                            ▼
                                        SQLite attendance.db  (encodings only, no photos)
```

Liveness uses the **eye aspect ratio (EAR)** from facial landmarks: eyes closing then re-opening within
the scan counts as a blink and is valid for 8 seconds.

## Tech stack

- **Backend:** Python, Flask, face_recognition (dlib), NumPy, Pillow, SQLite
- **Frontend:** HTML, CSS, vanilla JavaScript (getUserMedia, Canvas, fetch)

## API

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/register` | Register a person (requires `consent: true`) |
| POST | `/api/recognize` | Recognise faces in a frame, mark attendance |
| GET | `/api/people` | Registered people + attendance % |
| DELETE | `/api/people/<id>` | Delete a person and their data |
| GET | `/api/records?date=` | Attendance for a day |
| GET | `/api/stats` | Dashboard numbers and 7-day series |
| POST | `/api/manual` | Manual mark |
| DELETE | `/api/attendance/<id>?date=` | Undo a mark |
| GET/POST | `/api/settings` | Late-after time |
| GET | `/api/export.csv` | CSV (`?date=` or `?all=1`) |
| POST | `/api/wipe` | Delete everything |

## Ethics and privacy by design

| Principle | How it's done |
|---|---|
| Informed consent | Registration blocked without consent; timestamp stored and shown |
| Data minimisation | Only a 128-number template is stored, **never photos** |
| Local only | Server binds to 127.0.0.1; no external calls |
| Right to erasure | Per-person delete and "Delete ALL data" |
| Fairness fallback | Manual attendance for anyone who opts out or isn't recognised |
| Anti-spoofing | Optional blink-based liveness check |

## Known limitations / future work

- Blink detection is a basic liveness check; a video replay could still fool it. Next: depth or challenge-response checks.
- Accuracy depends on lighting and camera angle; test with a diverse group before real deployment.
- No login yet: add an admin PIN and encrypt `attendance.db` at rest.
- Absence alerts by email or Telegram, and multi-camera support.

## Troubleshooting

- **No module named face_recognition**: you're outside the venv. Run `venv\Scripts\activate` first.
- **Please install face_recognition_models**: `pip install "setuptools<81" face_recognition_models`.
- **Camera shows a wall or is blank**: pick the right camera in Chrome's camera settings and close Zoom/DroidCam.
- **Wrong matches**: lower `TOLERANCE` in `app.py` (0.5 → 0.45) and register in the demo lighting.
- **Blink not detected**: face the camera, good light, blink deliberately; raise `EAR_CLOSED` slightly (0.21 → 0.24).

## Team

**CodeTitans**: add your names here.
