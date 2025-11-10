# app.py
# Single-file FastAPI + Socket.IO server for alphabet ASL detection.
# Python 3.10 compatible.
# Dependencies: fastapi, "python-socketio[asgi]", uvicorn, opencv-python, mediapipe, numpy

import base64
import cv2
import numpy as np
import asyncio
import time
from typing import Dict, Any
import os

import socketio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Begin: HandTrackingModule copied (no logic changed) ---
import mediapipe as mp

# Environment Variables
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

print(f"[init] PORT={PORT}")
print(f"[init] DEBUG={DEBUG}")
print(f"[init] CORS_ORIGINS={ALLOWED_ORIGINS}")


class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        # Note: Keep the original constructor params; Mediapipe Hands object:
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True) :
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks :
            for handLand in self.result.multi_hand_landmarks :
                if draw :
                    self.mpDraw.draw_landmarks(img, handLand, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo = 0, draw = True) :
        PosList = []
        if self.result.multi_hand_landmarks :
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                PosList.append([id, cx, cy])
                
                if draw :
                    cv2.circle(img, (cx,cy), 10, (255,255,255), cv2.FILLED)
            
        return PosList
# --- End: HandTrackingModule ---

# Create one detector instance to reuse across frames (heavy to re-create)
_detector = handDetector(detectionCon=0.5, trackCon=0.5)

# Lock to serialize access to Mediapipe (MediaPipe can be sensitive to concurrent calls)
_detection_lock = asyncio.Lock()

# --- Detection / classification logic (kept intact from your script) ---
def classify_from_poslist(posList: list) -> str:
    """
    Accepts posList as returned by findPosition and returns the detected letter (or "" if none).
    This function mirrors the original detection decision tree/logic exactly.
    """
    result = ""
    fingers = []

    finger_mcp = [5,9,13,17]
    finger_dip = [6,10,14,18]
    finger_pip = [7,11,15,19]
    finger_tip = [8,12,16,20]

    for id in range(4):
        try:
            if(posList[finger_tip[id]][1]+ 25  < posList[finger_dip[id]][1] and posList[16][2]<posList[20][2]):
                fingers.append(0.25)
            elif(posList[finger_tip[id]][2] > posList[finger_dip[id]][2]):
                fingers.append(0)
            elif(posList[finger_tip[id]][2] < posList[finger_pip[id]][2]): 
                fingers.append(1)
            elif(posList[finger_tip[id]][1] > posList[finger_pip[id]][1] and posList[finger_tip[id]][1] > posList[finger_dip[id]][1]): 
                fingers.append(0.5)
        except:
            continue

    if len(fingers) != 4:
        return ""

    try:
        if(posList[3][2] > posList[4][2]) and (posList[3][1] > posList[6][1])and (posList[4][2] < posList[6][2]) and fingers.count(0) == 4:
            result = "A"
            
        elif(posList[3][1] > posList[4][1]) and fingers.count(1) == 4:
            result = "B"
        
        elif(posList[3][1] > posList[6][1]) and fingers.count(0.5) >= 1 and (posList[4][2]> posList[8][2]):
            result = "C"
            
        elif(fingers[0]==1) and fingers.count(0) == 3 and (posList[3][1] > posList[4][1]):
            result = "D"
        
        elif (posList[3][1] < posList[6][1]) and fingers.count(0) == 4 and posList[12][2]<posList[4][2]:
            result = "E"

        elif (fingers.count(1) == 3) and (fingers[0]==0) and (posList[3][2] > posList[4][2]):
            result = "F"

        elif(fingers[0]==0.25) and fingers.count(0) == 3:
            result = "G"

        elif(fingers[0]==0.25) and(fingers[1]==0.25) and fingers.count(0) == 2:
            result = "H"
        
        elif (posList[4][1] < posList[6][1]) and fingers.count(0) == 3:
            if (len(fingers)==4 and fingers[3] == 1):
                result = "I"
        
        elif (posList[4][1] < posList[6][1] and posList[4][1] > posList[10][1] and fingers.count(1) == 2):
            result = "K"
            
        elif(fingers[0]==1) and fingers.count(0) == 3 and (posList[3][1] < posList[4][1]):
            result = "L"
        
        elif (posList[4][1] < posList[16][1]) and fingers.count(0) == 4:
            result = "M"
        
        elif (posList[4][1] < posList[12][1]) and fingers.count(0) == 4:
            result = "N"
            
        elif (posList[4][1] > posList[12][1]) and posList[4][2]<posList[6][2] and fingers.count(0) == 4:
            result = "T"

        elif (posList[4][1] > posList[12][1]) and posList[4][2]<posList[12][2] and fingers.count(0) == 4:
            result = "S"

        
        elif(posList[4][2] < posList[8][2]) and (posList[4][2] < posList[12][2]) and (posList[4][2] < posList[16][2]) and (posList[4][2] < posList[20][2]):
            result = "O"
        
        elif(fingers[2] == 0)  and (posList[4][2] < posList[12][2]) and (posList[4][2] > posList[6][2]):
            if (len(fingers)==4 and fingers[3] == 0):
                result = "P"
        
        elif(fingers[1] == 0) and (fingers[2] == 0) and (fingers[3] == 0) and (posList[8][2] > posList[5][2]) and (posList[4][2] < posList[1][2]):
            result = "Q"
            
        elif(posList[8][1] < posList[12][1]) and (fingers.count(1) == 2) and (posList[9][1] > posList[4][1]):
            result = "R"
            
        elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2] and (posList[8][1] - posList[11][1]) <= 50):
            result = "U"
            
        elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 2 and posList[3][2] > posList[4][2]):
            result = "V"
        
        elif (posList[4][1] < posList[6][1] and posList[4][1] < posList[10][1] and fingers.count(1) == 3):
            result = "W"
        
        elif (fingers[0] == 0.5 and fingers.count(0) == 3 and posList[4][1] > posList[6][1]):
            result = "X"
        
        elif(fingers.count(0) == 3) and (posList[3][1] < posList[4][1]):
            if (len(fingers)==4 and fingers[3] == 1):
                result = "Y"
    except:
        return ""

    return result

async def detect_and_classify_image(bgr_frame: np.ndarray) -> Dict[str, Any]:
    """
    Runs the Mediapipe detection and classification synchronously inside a background thread.
    Returns a dict with 'label' and 'score' (score is simple confidence: 1.0 if non-empty label else 0.0).
    """
    # Use a lock to protect the Mediapipe detector and serialize calls.
    async with _detection_lock:
        # Run the CPU-bound detection in a thread to avoid blocking the event loop
        def _work(img: np.ndarray) -> Dict[str, Any]:
            img_proc = _detector.findHands(img, draw=False)
            posList = _detector.findPosition(img_proc, draw=False)
            label = classify_from_poslist(posList)
            score = 1.0 if label else 0.0
            return {"label": label, "score": score, "posList_len": len(posList)}
        result = await asyncio.to_thread(_work, bgr_frame)
    return result

# --- Socket.IO + FastAPI wiring ---
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI(title="ASL Alphabet Detection - Socket.IO")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
# Apply CORS like Node.js express()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount the socket.io ASGI app as the root app when using uvicorn:
# uvicorn app:socket_app --host 0.0.0.0 --port 8000

# Basic health route
@app.get("/healthz")
async def health():
    return JSONResponse({"status": "ok", "service": "ASL-alphabet-socketio"})

@sio.event
async def connect(sid, environ):
    print(f"[socket] Client connected: sid={sid}")
    await sio.emit("connected", {"message": "connected"}, to=sid)

@sio.event
async def disconnect(sid):
    print(f"[socket] Client disconnected: sid={sid}")

@sio.event
async def frame(sid, data):
    try:
        print(f"[frame] Received frame from sid={sid}")

        image_b64 = data.get("image")
        if not image_b64:
            print("[warn] Received empty frame payload")
            await sio.emit("prediction", {"label": "", "score": 0.0}, to=sid)
            return

        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            print("[warn] Failed to decode image")
            await sio.emit("prediction", {"label": "", "score": 0.0}, to=sid)
            return

        result = await detect_and_classify_image(img)

        print(f"[prediction] sid={sid}, label={result['label']}, score={result['score']}, posList_len={result['posList_len']}")

        await sio.emit("prediction", {
            "label": result["label"],
            "score": result["score"]
        }, to=sid)

    except Exception as e:
        print(f"[error] Exception in frame handler: {e}")
        await sio.emit("prediction", {"label": "", "score": 0.0}, to=sid)
