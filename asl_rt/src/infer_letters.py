# src/infer_letters.py
import cv2, pickle, collections, numpy as np
import mediapipe as mp
from features import flatten_landmarks

MODEL = "models/letters_sklearn.pkl"
SMOOTH_N = 7
CONF_THRESH = 0.55

with open(MODEL, "rb") as f:
    clf = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def majority(seq):
    return max(set(seq), key=seq.count) if seq else ""

def main():
    cap = cv2.VideoCapture(0)
    hist = collections.deque(maxlen=SMOOTH_N)
    text_buf = ""

    with mp_hands.Hands(model_complexity=1, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            pred, prob = "", 0.0
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                feats = flatten_landmarks(lm.landmark, w, h).reshape(1, -1)   # 42-D raw xy
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(feats)[0]
                    j = int(np.argmax(probs)); prob = float(probs[j])
                    if prob >= CONF_THRESH: pred = clf.classes_[j]
                else:
                    pred = clf.predict(feats)[0]; prob = 1.0 if pred else 0.0

                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            hist.append(pred if pred else "")
            if len(hist) == SMOOTH_N:
                m = majority(list(hist))
                if m and (not text_buf or text_buf[-1] != m):
                    text_buf += " " if m.lower() in ("space", "blank", "_") else m

            cv2.putText(frame, f"Pred: {pred} ({prob:.2f})", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Text: {text_buf[-60:]}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, "[q] quit  [c] clear  [b] backspace", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            cv2.imshow("ASL Letters (Real-Time)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k == ord('c'): text_buf = ""
            if k == ord('b') and text_buf: text_buf = text_buf[:-1]

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
