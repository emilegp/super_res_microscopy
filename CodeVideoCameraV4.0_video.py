import numpy as np
import matplotlib as plt
import pycromanager
from pycromanager import Core
import time
from typing import Any
from PIL import Image
import pylablib as pll
from pylablib.devices import Thorlabs
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE, TLCameraError

pll.par["devices/dlls/thorlabs_tlcam"] = r"C:\Users\marie\Documents\Scientific Camera Interfaces\SDK\Python Toolkit\dlls\64_lib"

#Numéro de série de la caméra
print(Thorlabs.list_cameras_tlcam())

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
            print("Caméra armée et prête pour capturer des images.")
            
            output_tiff_path = r"C:\Users\marie\Documents\GitHub\super_res_microscopy\video_output.tiff"

            # Capture and save images
            images = []  # List to store frames for TIFF
            for picture_number in range(100):  # Capture 100 frames
                cam1.issue_software_trigger()
                time.sleep(1 / nombre_image_par_seconde)  # Maintain frame rate

                frame = cam1.get_pending_frame_or_null()
                if frame is not None:
                    print(f"Image successfully captured: {frame.frame_count}")
                    # Convert the image buffer to a NumPy array
                    image = np.array(frame.image_buffer, dtype=np.uint8).reshape(
                        (cam1.image_height_pixels, cam1.image_width_pixels)
                    )
                    # Add the image to the list as a Pillow Image
                    images.append(Image.fromarray(image))
                else:
                    print("No image captured.")

            # Save all frames as a multi-page TIFF
            if images:
                images[0].save(
                    output_tiff_path,
                    save_all=True,
                    append_images=images[1:],  # Save as multi-page TIFF
                    compression="tiff_deflate"  # Optional: Apply compression
                )
                print(f"Multi-page TIFF saved at: {output_tiff_path}")
            else:
                print("No frames to save.")

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