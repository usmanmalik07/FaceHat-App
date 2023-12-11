import numpy as np
import cv2

class FaceHatApp:
    def __init__(self):
        self.hat_image = cv2.imread('C:/Users/DELL/Desktop/FaceHat App/hat.png', cv2.IMREAD_UNCHANGED)
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.capture_image = False
        self.captured_frame = None

    def put_hat(self, frame, x, y, w, h):
        hat_resized = cv2.resize(self.hat_image, (w, h))
        roi = frame[y:y+h, x:x+w]
        hat_color = hat_resized[:, :, 0:3]
        alpha_channel = hat_resized[:, :, 3] / 255.0
        alpha_channel = cv2.resize(alpha_channel, (w, h))
        alpha_channel = (alpha_channel * 255).astype(np.uint8)
        roi_bg = cv2.bitwise_and(roi, roi, mask=(1 - alpha_channel))
        roi_fg = cv2.bitwise_and(hat_color, hat_color, mask=alpha_channel)
        merged_hat = cv2.add(roi_bg, roi_fg)
        frame[y:y+h, x:x+w] = merged_hat

    def on_button_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.capture_image = True

    def main(self):
        cv2.namedWindow('Hat on Head')
        cv2.setMouseCallback('Hat on Head', self.on_button_click)

        while True:
            ret, frame = self.cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                y = max(0, min(y - 150, frame.shape[0] - h))
                self.put_hat(frame, x, y, w, h)

            cv2.imshow('Hat on Head', frame)

            if self.capture_image:
                self.captured_frame = frame.copy()
                self.capture_image = False

            if self.captured_frame is not None:
                cv2.imshow('Captured Image', self.captured_frame)
                self.capture_image = False

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceHatApp()
    app.main()
