import cv2

# モデルの読み込み
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

def detect_smile(gray, face):
    roi_gray = gray[face[1] + face[3]//2:face[1] + face[3], face[0]:face[0] + face[2]]
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
    return len(smiles) > 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = (x, y, w, h)

        smile_detected = detect_smile(gray, face)

        # 表情に応じて画像を選択
        if smile_detected:
            expression = 'happy'
            effect_image = cv2.imread('images/smile.png', cv2.IMREAD_UNCHANGED)
        else:
            expression = 'neutral'
            effect_image = cv2.imread('images/neutral_effect.png', cv2.IMREAD_UNCHANGED)

        # 画像サイズを顔のサイズにリサイズ
        effect_image = cv2.resize(effect_image, (w, h))

        # アルファブレンディングで画像を重ねる
        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = effect_image[:, :, c] * (effect_image[:, :, 3] / 255.0) + \
                                      frame[y:y+h, x:x+w, c] * (1.0 - effect_image[:, :, 3] / 255.0)

    cv2.imshow('Expression Effect', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()