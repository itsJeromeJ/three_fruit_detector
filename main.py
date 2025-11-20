from ultralytics import YOLO
import cv2
try:
    img=cv2.imread(r"C:\Users\jerom\OneDrive\Desktop\pratices\API\assessment\dl\testfriut\test\download (2).jpg")
    
    try:
        model=YOLO(r'C:\Users\jerom\OneDrive\Desktop\pratices\API\assessment\dl\yolo11m.pt')
        model.predict(img, save=True, show_conf = True, show_labels=True)
    except Exception as e:
        print(f"error on model:{e}")

except Exception as e:
    print(f"error on image: {e}")
