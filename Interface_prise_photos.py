import cv2
import tkinter as tk
from tkinter import messagebox
import os

def Prendre_Photo():
    cam_gauche = cv2.VideoCapture(0)  
    cam_droite = cv2.VideoCapture(1)

    nom_enregistrement = "Photos prises"
    os.makedirs(nom_enregistrement, exist_ok=True)
    messagebox.showinfo("Instructions", "Placezle damier en 5 positions différentes.\n Et appuyez sur ESPACE après chaque position.")

    _, frame_gauche = cam_gauche.read()
    _, frame_droite = cam_droite.read()

    key = cv2.waitKey(1)
    if key == 32:  # Touche ESPACE
        cv2.imwrite(f"{nom_enregistrement}/photo_left.jpg", frame_gauche)
        cv2.imwrite(f"{nom_enregistrement}/photo_right.jpg", frame_droite)
    
    cam_gauche.release()
    cam_droite.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Terminé,les images ont été enregistrées avec succès !")

root = tk.Tk()
root.title("Calibration Stéréo")
bouton_calibrate = tk.Button(root, text="Lancer la calibration", command=Prendre_Photo())
bouton_calibrate.pack(pady=20)
root.mainloop()