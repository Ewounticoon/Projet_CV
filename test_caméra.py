import cv2
import tkinter as tk
from tkinter import messagebox
import os

def capture_images():
    cam_left = cv2.VideoCapture(0)  # Première caméra

    if not cam_left.isOpened():
        messagebox.showerror("Erreur", "Impossible d'ouvrir les caméras")
        return
    
    save_path = "calibration_images"
    os.makedirs(save_path, exist_ok=True)
    
    messagebox.showinfo("Instructions", "Placez le banc face au damier en 5 positions différentes.\nAppuyez sur ESPACE après chaque position.")
    
    for i in range(5):
        messagebox.showinfo("Position {}".format(i+1), "Placez le damier à la position {} et appuyez sur ESPACE".format(i+1))
        while True:
            ret_left, frame_left = cam_left.read()
            if not ret_left:
                messagebox.showerror("Erreur", "Problème de capture vidéo")
                return
            
            cv2.imshow("Prévisualisation - Appuyez sur ESPACE pour capturer", ret_left)
            
            key = cv2.waitKey(1)
            if key == 32:  # Touche ESPACE
                cv2.imwrite(f"{save_path}/left_{i}.jpg", frame_left)
                break
    
    cam_left.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Terminé", "5 images capturées avec succès !")

# Interface utilisateur
root = tk.Tk()
root.title("Calibration 1 caméra")
btn_calibrate = tk.Button(root, text="Lancer la Calibration", command=capture_images, font=("Arial", 14))
btn_calibrate.pack(pady=20)
root.mainloop()



