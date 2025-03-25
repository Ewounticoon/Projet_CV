import cv2
import tkinter as tk
from tkinter import messagebox
import os

def capture_images():
    cam_left = cv2.VideoCapture(0)  # Première caméra
    cam_right = cv2.VideoCapture(1)  # Deuxième caméra
    
    if not cam_left.isOpened() or not cam_right.isOpened():
        messagebox.showerror("Erreur", "Impossible d'ouvrir les caméras")
        return
    
    save_path = "calibration_images"
    os.makedirs(save_path, exist_ok=True)
    
    messagebox.showinfo("Instructions", "Placez le banc face au damier en 5 positions différentes.\nAppuyez sur ESPACE après chaque position.")
    
    for i in range(5):
        messagebox.showinfo("Position {}".format(i+1), "Placez le damier à la position {} et appuyez sur ESPACE".format(i+1))
        while True:
            ret_left, frame_left = cam_left.read()
            ret_right, frame_right = cam_right.read()
            if not ret_left or not ret_right:
                messagebox.showerror("Erreur", "Problème de capture vidéo")
                return
            
            combined = cv2.hconcat([frame_left, frame_right])
            cv2.imshow("Prévisualisation - Appuyez sur ESPACE pour capturer", combined)
            
            key = cv2.waitKey(1)
            if key == 32:  # Touche ESPACE
                cv2.imwrite(f"{save_path}/left_{i}.jpg", frame_left)
                cv2.imwrite(f"{save_path}/right_{i}.jpg", frame_right)
                break
    
    cam_left.release()
    cam_right.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Terminé", "10 images capturées avec succès !")

# Interface utilisateur
root = tk.Tk()
root.title("Calibration Stéréo")
btn_calibrate = tk.Button(root, text="Lancer la Calibration", command=capture_images, font=("Arial", 14))
btn_calibrate.pack(pady=20)
root.mainloop()


def Take_Photo():
    cam_left = cv2.VideoCapture(0)  
    cam_right = cv2.VideoCapture(1)

    save_path = "Taken_photos"
    os.makedirs(save_path, exist_ok=True)
    messagebox.showinfo("Instructions", "Placez le banc face au damier en 5 positions différentes.\nAppuyez sur ESPACE après chaque position.")

    ret_left, frame_left = cam_left.read()
    ret_right, frame_right = cam_right.read()
    if not ret_left or not ret_right:
        messagebox.showerror("Erreur", "Problème de capture vidéo")
        return

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
