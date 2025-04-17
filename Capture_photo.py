import cv2
import tkinter as tk
import os
from tkinter import messagebox


def Take_Photo():
    cam_left = cv2.VideoCapture(1)  
    cam_right = cv2.VideoCapture(2)

    save_path = "Taken_photos"
    os.makedirs(save_path, exist_ok=True)
    messagebox.showinfo("Instructions", "Placez le banc face au damier en 5 positions différentes.\nAppuyez sur ESPACE après chaque position.")
    while True :
        ret_left, frame_left = cam_left.read()
        ret_right, frame_right = cam_right.read()
        if not ret_left or not ret_right:
            messagebox.showerror("Erreur", "Problème de capture vidéo")
            break

        combined = cv2.hconcat([frame_left, frame_right])
        cv2.imshow("Prévisualisation - Appuyez sur ESPACE pour capturer", combined)

        key = cv2.waitKey(1)
        if key == 32:  # Touche ESPACE
            cv2.imwrite(f"{save_path}/photo_left.jpg", frame_left)
            cv2.imwrite(f"{save_path}/photo_right.jpg", frame_right)
        
    cam_left.release()
    cam_right.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Terminé", "les images ont été capturées avec succès !")

root = tk.Tk()
root.title("Calibration Stéréo")
btn_calibrate = tk.Button(root, text="Lancer la Calibration", command=Take_Photo(), font=("Arial", 14))
btn_calibrate.pack(pady=20)
root.mainloop()