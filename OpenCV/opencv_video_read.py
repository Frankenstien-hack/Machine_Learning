import cv2

cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue

    cv2.imshow("Video frame", frame)
    cv2.imshow("Gray frame", grayframe)
    keypressed = cv2.waitKey(1) & 0xff
    
    if keypressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()