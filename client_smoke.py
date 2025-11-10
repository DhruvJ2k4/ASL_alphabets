# client_smoke.py
# Test client to stream webcam frames to the ASL FastAPI+Socket.IO backend.
# Requires: python-socketio, opencv-python

import socketio
import cv2
import base64

# Adjust if your backend runs on a different host or port
SERVER_URL = "http://localhost:8000"

sio = socketio.Client()

@sio.event
def connect():
    print("[client] connected to server")

@sio.event
def disconnect():
    print("[client] disconnected")

@sio.on("prediction")
def on_prediction(data):
    # data = {"label": "A/B/C/etc", "score": float}
    print(f"[prediction] label={data.get('label')}  score={data.get('score')}")

def main():
    sio.connect(SERVER_URL, transports=["websocket"])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access webcam")
        return
    
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Encode frame as JPEG
        success, encoded_image = cv2.imencode(".jpg", frame)
        if not success:
            continue
        
        b64_image = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
        # Send to server
        sio.emit("frame", {"image": b64_image})

        # Display local preview window (optional)
        cv2.imshow("client webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sio.disconnect()

if __name__ == "__main__":
    main()
