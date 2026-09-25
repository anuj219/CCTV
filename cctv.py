import cv2

url = "rtsp://admin:admin123@192.168.29.62:554/Streaming/Channels/301"
cap = cv2.VideoCapture(url)
count = 1

if not cap.isOpened():
    print("Failed to open stream")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(r"C:\Users\anujv\Desktop\Programming\Codes\python\Face Recognition\NeytraN\cctv\known_faces\hiyansh\5.jpg", frame)
        print("Success — check test_frame.jpg")
    cap.release()