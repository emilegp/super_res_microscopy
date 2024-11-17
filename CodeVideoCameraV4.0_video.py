import numpy as np
import matplotlib as plt
import pycromanager
from pycromanager import Core
import time
import os
import cv2
from typing import Any



import pylablib as pll
pll.par["devices/dlls/thorlabs_tlcam"] = r"C:\Users\marie\Documents\Scientific Camera Interfaces\SDK\Python Toolkit\dlls\64_lib"
from pylablib.devices import Thorlabs


#Numéro de série de la caméra
print(Thorlabs.list_cameras_tlcam())

import pylablib
from pylablib.devices import Thorlabs
import thorlabs_tsi_sdk 
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE, TLCameraError

try:
    # Initialiser le SDK
    sdk = TLCameraSDK()
    print("SDK initialisé avec succès.")

    # Découvrir les caméras disponibles
    available_cameras = sdk.discover_available_cameras()
    print(f"Caméras disponibles : {available_cameras}")

    if not available_cameras:
        print("Aucune caméra trouvée. Vérifiez la connexion.")
        sdk.dispose()
        exit()

    # Essayer d'ouvrir la première caméra disponible
    try:
        with sdk.open_camera(available_cameras[0]) as cam1:
            print(f"Caméra connectée avec succès : {available_cameras[0]}")

            # Configurer les paramètres de la caméra
            cam1.exposure_time_us = 25000  # Exemple de temps d'exposition de 50 ms (tester sur Thorlab)
            cam1.black_level=5
            cam1.gain=0
            cam1.frames_per_trigger_zero_for_unlimited = 1  # Mode de capture continue : 0 ; Mode de capture ponctuel : 1
            cam1.arm(2)  # Préparez la caméra avec 2 tampons
            nombre_image_par_seconde=10
            output_path = fr"C:\Users\marie\Documents\GitHub\super_res_microscopy\image2.avi"
            print("Caméra armée et prête pour capturer des images.")
            
            #Définir les dimensions de la vidéo
            frame_width = cam1.image_width_pixels
            frame_height = cam1.image_height_pixels
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, nombre_image_par_seconde, (frame_width, frame_height), isColor=False)


            for picture_number in range (100):
                # Commencer à capturer des imagesc 
                cam1.issue_software_trigger()
                time.sleep(1/nombre_image_par_seconde)
                print("Capture d'images en cours...")
                frame = cam1.get_pending_frame_or_null()
                #print(frame.frame_count)

                if frame is not None:
                    print(f"Image capturée avec succès : {frame.frame_count}")
                    image = frame.image_buffer  # Assurez-vous que cet attribut est correct
                    #cv2.imshow('Image Capturee', image)  # Affichez l'image
                    #cv2.waitKey(0)  # Attendez une touche pour fermer la fenêtre

                    # Enregistrer l'image en format video
                    image = np.array(frame.image_buffer, dtype=np.uint8).reshape((frame_height, frame_width))
                    out.write(image)
                    #filename = fr"C:\Users\User\Desktop\Polyautomne2024\Laboratoire Lucien\Test_pictures\image_videopourlucien.tiff"
                    #cv2.imwrite(filename, image)
                    #print(f"Image enregistrée sous : {filename}")
                else:
                    print("Aucune image capturée.")
            

            # Désarmer et fermer la caméra après utilisation
            cam1.disarm()
            print("Caméra désarmée et fermée.")


    except TLCameraError as e:
        print(f"Erreur lors de la connexion de la caméra : {e}")
        print("Vérifiez que la caméra est correctement connectée et non utilisée par un autre programme.")

    finally:
        # Assurez-vous que le SDK est libéré pour libérer les ressources
        sdk.dispose()

except TLCameraError as e:
    print(f"Erreur SDK : {e}")

finally:
    # Nettoyer le SDK correctement
    if 'sdk' in locals():
        sdk.dispose()