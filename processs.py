import sys
import cv2
import ultralytics
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt


############### Model Path ###############
model = ultralytics.YOLO(r"runs\\detect\\train7\\weights\\best.pt")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi("design.ui", self)

        self.statusEdit.setReadOnly(True)
        self.statusEdit.setText("Starting...")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # lower ms

        # Timer (frame update)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.statusEdit.setText("Camera read failed!")
            return

        frame = cv2.resize(frame, (1920, 1080)) 

        results = model(frame, verbose=False)
        annotated = results[0].plot()  # BGR numpy image

        r = results[0]
        names = r.names

        priority = ["No Mask", "Mask Incorrect", "Mask"]
        
        status = "No detection"
        
        if len(r.boxes) > 0:
        
            detections = []
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = names[cls_id]
        
                detections.append((class_name, conf))
        
            for p in priority:
                for det in detections:
                    if det[0] == p:
                        status = f"{det[0]} {det[1]*100:.1f}%"
                        break
                if status != "No detection":
                    break
        
        self.statusEdit.setText(status)
        
        if "No Mask" in status:
            self.statusEdit.setStyleSheet("""
                background-color: #2b0000;
                color: red;
                border: 2px solid red;
                font-weight: bold;
            """)

        elif "Mask Incorrect" in status:
            self.statusEdit.setStyleSheet("""
                background-color: #2b1a00;
                color: orange;
                border: 2px solid orange;
                font-weight: bold;
            """)
        
        elif "Mask" in status:
            self.statusEdit.setStyleSheet("""
                background-color: #002b00;
                color: lime;
                border: 2px solid lime;
                font-weight: bold;
            """)
        
        else:
            self.statusEdit.setStyleSheet("""
                background-color: #1e1e1e;
                color: gray;
                border: 1px solid #444;
            """)
        
        
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        pix = pix.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(pix)

    def closeEvent(self, event):
        
        try:
            self.timer.stop()
        except:
            pass
        if self.cap is not None:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
