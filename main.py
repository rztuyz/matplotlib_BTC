import cv2

detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320)
)
def main():
    cap = cv2.VideoCapture(0)
    while True :
        ret,frame = cap.read()
        if not ret:
            break
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        faces = detector.detect(frame)[1]

        if faces is not None:
            for f in faces:
                x, y, w, h = map(int, f[:4])
                score = f[14]#Уверенность модели
                
                if score < 0.7:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{score:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x, max(0, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 225),
                    1,
                    cv2.LINE_AA
                )
                
        cv2.imshow("YuNet", frame)
        if cv2.waitKey(1) == 27:
            break              
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
