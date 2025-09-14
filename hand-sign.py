# hand_sign_fast_switch.py
import cv2, time, math, threading, tempfile, os
import mediapipe as mp
from gtts import gTTS
import pygame

# === Config ===
YOUR_NAME = "Rifa'i"
NAME_PHON = "ri fa i"
WORDS = {"HALO":"Halo","PERKENALKAN":"Perkenalkan","NAMA":"Nama Saya","YOUR_NAME":YOUR_NAME,"TERIMAKASIH":"Terima Kasih"}
FONT_SCALE, FONT_THICK = 0.9, 2
TEXT_POS = (20, 40)
NEED_STABLE = 8
MIN_TTS_GAP = 0.35
COOLDOWN_SAME = 1.0 
THUMB_SLACK = 0.02
MARGIN_EXT_Y, MARGIN_CLOSED_Y = 0.02, 0.008

# === TTS (ID) ===
_pyg_ok, _tts_lock, _last_tts = False, threading.Lock(), 0.0
def _ensure_pyg():
    global _pyg_ok
    if not _pyg_ok:
        pygame.mixer.init(); _pyg_ok = True
def speak_async(text):
    def run():
        global _last_tts
        with _tts_lock:
            if time.time() - _last_tts < MIN_TTS_GAP: return
            utt = text.replace(YOUR_NAME, NAME_PHON).replace("’"," ").replace("'"," ")
            _ensure_pyg()
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    gTTS(text=utt, lang="id").save(f.name); path = f.name
                pygame.mixer.music.load(path); pygame.mixer.music.play()
                while pygame.mixer.music.get_busy(): time.sleep(0.03)
                os.unlink(path)
            except Exception as e:
                print("TTS error:", e)
            _last_tts = time.time()
    threading.Thread(target=run, daemon=True).start()

# === MediaPipe ===
mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
TIP = {"thumb":4,"index":8,"middle":12,"ring":16,"pinky":20}
PIP = {"index":6,"middle":10,"ring":14,"pinky":18}
IP = 3
def dist(a,b): return math.hypot(a.x-b.x, a.y-b.y)

# === Classify ===
def ext_non_thumb(l,f):
    tip, pip = l[TIP[f]], l[PIP[f]]
    return tip.y < pip.y - MARGIN_EXT_Y
def closed_non_thumb(l,f):
    tip, pip = l[TIP[f]], l[PIP[f]]
    return tip.y > pip.y + MARGIN_CLOSED_Y
def ext_thumb(l,hand):
    tipx, ipx = l[TIP["thumb"]].x, l[IP].x
    return tipx < ipx - THUMB_SLACK if hand=="Right" else tipx > ipx + THUMB_SLACK

def classify(l, hand):
    t = ext_thumb(l,hand)
    i = ext_non_thumb(l,"index")
    m = ext_non_thumb(l,"middle")
    r = ext_non_thumb(l,"ring")
    p = ext_non_thumb(l,"pinky")
    ci, cm, cr, cp = (closed_non_thumb(l,f) for f in ["index","middle","ring","pinky"])

    if p and (not t) and (not i) and (not m) and (not r): return "YOUR_NAME"
    if ci and cm and cr and cp and (not t):                return "TERIMAKASIH"
    if i and m and (not r) and (not p):                    return "PERKENALKAN"
    if t and (not i) and (not m) and (not r) and (not p):  return "NAMA"
    if i and m and r and p:                                 return "HALO"
    return "UNKNOWN"

# === Main ===
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Kamera tidak ditemukan."); return
    last_label, stable = "NONE", 0
    last_play = {}   # per-label cooldown
    show_text, show_until = "", 0.0

    with mp_hands.Hands(model_complexity=1, max_num_hands=2,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame,1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            label = "NONE"
            if res.multi_hand_landmarks and res.multi_handedness:
                for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    cand = classify(lms.landmark, hd.classification[0].label)
                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                    if cand != "UNKNOWN": label = cand; break

            stable = stable + 1 if label == last_label else 1
            last_label = label

            now = time.time()
            if (label in WORDS) and (stable >= NEED_STABLE) and (now - last_play.get(label, 0) >= COOLDOWN_SAME) and (now - _last_tts >= MIN_TTS_GAP):
                word = WORDS[label]
                show_text, show_until = word, now + (1.0 + 0.07*len(word))
                speak_async(word)
                last_play[label] = now

            if time.time() <= show_until and show_text:
                cv2.putText(frame, show_text, TEXT_POS, cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE, (0,255,0), FONT_THICK, cv2.LINE_AA)

            cv2.imshow("Hand Sign (Fast Switch)", frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
