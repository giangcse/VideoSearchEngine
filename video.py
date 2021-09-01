from PIL import Image
import pytesseract
import cv2
import os


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if (int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % int(int(cap.get(cv2.CAP_PROP_FPS))/3) == 0):
                width, height, channels = frame.shape
                if width > 1280:
                    frame = cv2.resize(
                        frame, (1280, int(1280 * height / width)))
                    width, height, channels = frame.shape
                # Chuyển thành ảnh xám
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Phân tách đen trắng
                gray = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                # Ghi tạm ảnh xuống ổ cứng để sau đó apply OCR
                filename = "{}.png".format(os.getpid())
                cv2.imwrite(filename, gray)
                # Load ảnh và apply nhận dạng bằng Tesseract OCR
                text = pytesseract.image_to_string(
                    Image.open(filename), lang='vie')
                # Xóa ảnh tạm sau khi nhận dạng
                os.remove(filename)
                print(text)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = 'videos/2.mp4'
    read_video(video_path)
