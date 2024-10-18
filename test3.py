import cv2

# モデルの読み込み
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

def mosaic(img, rect, size):
    # 顔領域の座標を取得
    (x, y, w, h) = rect

    # 顔を一旦切り出し、リサイズしてモザイク処理
    roi = img[y:y+h, x:x+w]
    roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_LINEAR)
    roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)

    # 元の画像にモザイク処理した顔を戻す
    img[y:y+h, x:x+w] = roi

    return img

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # 顔の領域にモザイク処理を施す
        frame = mosaic(frame, (x, y, w, h), size=10)

    cv2.imshow('Mosaic Effect', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()
