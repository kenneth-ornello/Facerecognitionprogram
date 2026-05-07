import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import csv
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import util

# ── Paths (everything stays on this machine – no cloud) ──────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
STUDENTS_JSON = os.path.join(BASE_DIR, 'students.json')
DB_DIR        = os.path.join(BASE_DIR, 'db')
LOG_PATH      = os.path.join(BASE_DIR, 'log.csv')

# ── CSV header ───────────────────────────────────────────────────────────────
CSV_HEADER = ['Roll No', 'Name', 'Division', 'Department', 'Timestamp', 'Status']

DEPARTMENTS = [
    "Computer Engineering",
    "Information Technology",
    "Electronics & Telecomm.",
    "Mechanical Engineering",

]
DIVISIONS = ["A", "B"]

# ── Theme colours ─────────────────────────────────────────────────────────────
BG_DARK   = "#1a1a2e"
BG_PANEL  = "#16213e"
FG_LIGHT  = "#dfe6e9"
BTN_GREEN = "#00b894"
BTN_RED   = "#e17055"
BTN_BLUE  = "#74b9ff"
BTN_AMBER = "#fdcb6e"
ENTRY_BG  = "#0f3460"


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_students() -> dict:
    if os.path.exists(STUDENTS_JSON):
        with open(STUDENTS_JSON, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_students(data: dict):
    with open(STUDENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def init_log():
    """Create the CSV with a header row if it doesn't already exist."""
    os.makedirs(os.path.dirname(LOG_PATH) or '.', exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_log(roll: str, name: str, division: str, department: str, status: str):
    """Append a single attendance row to the CSV (local file only)."""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([roll, name, division, department, ts, status])


def already_marked_today(roll: str, status: str) -> bool:
    """Return True if the same roll + status was already logged today."""
    today = datetime.date.today().isoformat()
    if not os.path.exists(LOG_PATH):
        return False
    with open(LOG_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if (row.get('Roll No', '').strip() == roll and
                    row.get('Status', '').strip() == status and
                    row.get('Timestamp', '').startswith(today)):
                return True
    return False


def purge_deepface_cache():
    """Remove .pkl cache files so newly registered faces are picked up."""
    for fname in os.listdir(DB_DIR):
        if fname.endswith('.pkl'):
            try:
                os.remove(os.path.join(DB_DIR, fname))
            except OSError:
                pass


# ── Main application ──────────────────────────────────────────────────────────

class App:
    def __init__(self):
        init_log()
        os.makedirs(DB_DIR, exist_ok=True)

        self.root = tk.Tk()
        self.root.geometry("1220x580+300+60")
        self.root.title("DBIT Face Attendance System")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        self._build_ui()
        self._start_webcam()

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Left: live feed
        self.cam_label = util.get_img_label(self.root)
        self.cam_label.place(x=10, y=10, width=700, height=520)

        # Right panel background
        panel = tk.Frame(self.root, bg=BG_PANEL, width=490, height=560)
        panel.place(x=720, y=10)

        tk.Label(panel, text="DBIT Attendance", bg=BG_PANEL, fg=BTN_BLUE,
                 font=("Helvetica bold", 20)).place(x=20, y=18)

        # Status box
        self.status_var = tk.StringVar(value="🟢  System ready")
        tk.Label(panel, textvariable=self.status_var, bg=BG_PANEL, fg=FG_LIGHT,
                 font=("Helvetica", 12), wraplength=440, justify='left'
                 ).place(x=20, y=58)

        # Separator
        tk.Frame(panel, bg="#2d3436", height=2, width=450).place(x=20, y=110)

        # Action buttons
        buttons = [
            ("✅  Mark Attendance",  BTN_GREEN, self.login,             135),
            ("🚪  Mark Exit",         BTN_RED,   self.logout,            215),
            ("➕  Register Student",  BTN_BLUE,  self.register_new_user, 295),
            ("📋  View Today's Log",  BTN_AMBER, self.show_log,          375),
        ]
        for text, colour, cmd, y in buttons:
            tk.Button(panel, text=text, bg=colour, fg='white',
                      activebackground='black', activeforeground='white',
                      command=cmd, height=2, width=26,
                      font=('Helvetica bold', 14), relief='flat',
                      cursor='hand2').place(x=20, y=y)

        # Footer
        tk.Label(panel, text="All data stored locally on this device.",
                 bg=BG_PANEL, fg="#636e72",
                 font=("Helvetica", 10)).place(x=20, y=520)

    # ── Webcam ─────────────────────────────────────────────────────────────────

    def _start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.latest_frame = None
        self._tick()

    def _tick(self):
        ret, frame = self.cap.read()
        if ret:
            self.latest_frame = frame
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
        self.cam_label.after(20, self._tick)

    # ── Attendance actions ─────────────────────────────────────────────────────

    def _record(self, status: str, greeting: str):
        if self.latest_frame is None:
            util.msg_box("Error", "Webcam not ready.")
            return

        self.status_var.set("🔍  Scanning face…")
        self.root.update()

        key = util.recognize(self.latest_frame, DB_DIR)

        if key == 'no_persons_found':
            util.msg_box('No Face Detected',
                         'No face found in frame.\n'
                         'Ensure your face is well-lit and centred.')
            self.status_var.set("⚠️  No face detected.")
            return

        if key == 'unknown_person':
            util.msg_box('Not Recognised',
                         'Face detected but not in the database.\n'
                         'Please register first.')
            self.status_var.set("❌  Unknown face detected.")
            return

        students = load_students()
        s        = students.get(key, {})
        roll     = s.get('roll', key)
        name     = s.get('name', key)
        division = s.get('division', 'N/A')
        dept     = s.get('department', 'N/A')

        # Prevent duplicate entries for the same session
        if already_marked_today(roll, status):
            util.msg_box('Already Recorded',
                         f'{name} ({roll}) is already marked "{status}" today.')
            self.status_var.set(f"ℹ️  {name} already marked today.")
            return

        # ── Write to local CSV ────────────────────────────────────────────────
        append_log(roll, name, division, dept, status)

        util.msg_box(greeting, f'{greeting}, {name}!\nRoll: {roll}  |  Div: {division}')
        self.status_var.set(f"✅  {greeting}, {name} ({roll}) — logged to CSV.")

    def login(self):
        self._record('present', 'Welcome')

    def logout(self):
        self._record('exit', 'Goodbye')

    # ── View today's log ───────────────────────────────────────────────────────

    def show_log(self):
        today  = datetime.date.today().isoformat()
        rows   = []

        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('Timestamp', '').startswith(today):
                        rows.append(row)

        win = tk.Toplevel(self.root)
        win.title(f"Attendance – {today}")
        win.geometry("900x420+300+200")
        win.configure(bg=BG_DARK)

        cols = CSV_HEADER
        tree = ttk.Treeview(win, columns=cols, show='headings', height=15)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=140, anchor='center')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview",
                         background=BG_PANEL, foreground=FG_LIGHT,
                         rowheight=28, fieldbackground=BG_PANEL,
                         font=('Helvetica', 11))
        style.configure("Treeview.Heading",
                         background=BG_DARK, foreground=BTN_BLUE,
                         font=('Helvetica bold', 11))

        for r in rows:
            tree.insert('', 'end', values=[r.get(c, '') for c in cols])

        sb = ttk.Scrollbar(win, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        sb.pack(side='right', fill='y', pady=10, padx=(0, 10))

        if not rows:
            tk.Label(win, text="No attendance records for today yet.",
                     bg=BG_DARK, fg=FG_LIGHT,
                     font=('Helvetica', 14)).place(relx=0.5, rely=0.5, anchor='center')

    # ── Registration window ────────────────────────────────────────────────────

    def register_new_user(self):
        if self.latest_frame is None:
            util.msg_box("Error", "Webcam not ready.")
            return

        snap = self.latest_frame.copy()   # capture the current frame

        win = tk.Toplevel(self.root)
        win.geometry("1220x580+320+100")
        win.title("Register New Student")
        win.configure(bg=BG_PANEL)

        # Show snapshot
        snap_label = util.get_img_label(win)
        snap_label.place(x=10, y=10, width=700, height=520)
        rgb    = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
        imgtk  = ImageTk.PhotoImage(Image.fromarray(rgb))
        snap_label.imgtk = imgtk
        snap_label.configure(image=imgtk)

        # ── Live re-capture button ────────────────────────────────────────────
        capture_holder = {'frame': snap}

        def retake():
            if self.latest_frame is not None:
                capture_holder['frame'] = self.latest_frame.copy()
                rgb2   = cv2.cvtColor(capture_holder['frame'], cv2.COLOR_BGR2RGB)
                imgtk2 = ImageTk.PhotoImage(Image.fromarray(rgb2))
                snap_label.imgtk = imgtk2
                snap_label.configure(image=imgtk2)
                status_lbl.config(text="📸  New photo captured.")

        # Form
        lbl_cfg   = dict(bg=BG_PANEL, fg=FG_LIGHT, font=('Helvetica', 14))
        entry_cfg = dict(font=('Helvetica', 13), width=22, relief='flat',
                         bg=ENTRY_BG, fg='white', insertbackground='white')

        fields = {}
        for key, label, y in [('roll', 'Roll Number', 60), ('name', 'Full Name', 130)]:
            tk.Label(win, text=label, **lbl_cfg).place(x=740, y=y)
            e = tk.Entry(win, **entry_cfg)
            e.place(x=740, y=y + 30, width=260)
            fields[key] = e

        tk.Label(win, text='Division', **lbl_cfg).place(x=740, y=210)
        div_var = tk.StringVar(value=DIVISIONS[0])
        ttk.Combobox(win, textvariable=div_var, values=DIVISIONS,
                     font=('Helvetica', 13), width=10,
                     state='readonly').place(x=740, y=240)

        tk.Label(win, text='Department', **lbl_cfg).place(x=740, y=290)
        dept_var = tk.StringVar(value=DEPARTMENTS[0])
        ttk.Combobox(win, textvariable=dept_var, values=DEPARTMENTS,
                     font=('Helvetica', 12), width=28,
                     state='readonly').place(x=740, y=320)

        status_lbl = tk.Label(win, text="", bg=BG_PANEL, fg=BTN_AMBER,
                               font=('Helvetica', 11), wraplength=260)
        status_lbl.place(x=740, y=375)

        # ── Accept ────────────────────────────────────────────────────────────
        def accept():
            roll = fields['roll'].get().strip().upper()
            name = fields['name'].get().strip()
            div  = div_var.get().strip()
            dept = dept_var.get().strip()

            if not roll or not name:
                util.msg_box('Error', 'Roll number and name are required.')
                return

            # Save face image locally (no cloud)
            img_path = os.path.join(DB_DIR, f'{roll}.jpg')
            face_img = util.preprocess_face(capture_holder['frame'])
            cv2.imwrite(img_path, face_img)

            # Update local students registry
            students       = load_students()
            students[roll] = {'roll': roll, 'name': name,
                              'division': div, 'department': dept}
            save_students(students)

            # Invalidate DeepFace cache so new face is picked up immediately
            purge_deepface_cache()

            util.msg_box('Registered!',
                         f'{name} ({roll}) registered.\n'
                         f'They will appear in the CSV the next time\n'
                         f'attendance is marked.')
            self.status_var.set(f"✅  Registered: {name} ({roll})")
            win.destroy()

        tk.Button(win, text='📸  Retake Photo', bg='#6c5ce7', fg='white',
                  activebackground='black', command=retake,
                  height=2, width=18, font=('Helvetica bold', 13),
                  relief='flat', cursor='hand2').place(x=740, y=415)

        tk.Button(win, text='✔  Confirm Registration', bg=BTN_GREEN, fg='white',
                  activebackground='black', command=accept,
                  height=2, width=24, font=('Helvetica bold', 14),
                  relief='flat', cursor='hand2').place(x=740, y=465)

        tk.Button(win, text='✖  Cancel', bg=BTN_RED, fg='white',
                  activebackground='black', command=win.destroy,
                  height=2, width=12, font=('Helvetica bold', 14),
                  relief='flat', cursor='hand2').place(x=1050, y=465)

    # ── Run ────────────────────────────────────────────────────────────────────

    def start(self):
        self.root.mainloop()
        if hasattr(self, 'cap'):
            self.cap.release()


if __name__ == "__main__":
    App().start()