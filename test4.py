import cv2
import pygame.mixer
import time

# モデルの読み込み
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Pygame初期化
def init_sound():
    pygame.mixer.init()

# 音を再生する関数
def play_sound(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play(1)

# 笑顔検出
def detect_smile(gray, face):
    roi_gray = gray[face[1] + face[3]//2:face[1] + face[3], face[0]:face[0] + face[2]]
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
    return len(smiles) > 0

def main():
    cap = cv2.VideoCapture(0)
    init_sound()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = (x, y, w, h)

            smile_detected = detect_smile(gray, face)

            # 笑顔が検出されたら音を再生
            if smile_detected:
                play_sound("きらきら輝く3.mp3")

            # 顔の描画
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Smile Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
