import os
import tkinter as tk
from tkinter import messagebox
from deepface import DeepFace

# UI Helpers for main.py
def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
        window,
        text=text,
        activebackground="black",
        activeforeground="white",
        fg=fg,
        bg=color,
        command=command,
        height=2,
        width=20,
        font=('Helvetica bold', 20)
    )
    return button

def get_img_label(window):
    label = tk.Label(window)
    return label

def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label

def get_entry_text(window):
    inputtxt = tk.Text(window, height=2, width=15, font=("Arial", 32))
    return inputtxt

def msg_box(title, description):
    messagebox.showinfo(title, description)

# Recognition Logic
def recognize(img, db_path):
    try:
        # DeepFace scans the db folder for matches to the current webcam frame
        results = DeepFace.find(img_path=img,
                                db_path=db_path,
                                enforce_detection=False,
                                model_name='VGG-Face')

        if len(results) > 0 and not results[0].empty:
            # Extract name from the image path found in the results
            file_path = results[0]['identity'][0]
            name = os.path.splitext(os.path.basename(file_path))[0]
            return name
        else:
            return 'unknown_person'

    except Exception as e:
        print(f"DeepFace Recognition Error: {e}")
        return 'no_persons_found'