import cv2

close_value = True
n = 0
vidcap = cv2.VideoCapture(0)  # number is a index of camera which you have to choose
while close_value:
    succes, img = vidcap.read()
    if succes:
        cv2.imshow('Preview', img)

    k = cv2.waitKey(5)
    if k == 27:  #ESC key
        close_value = False
    elif k == ord('s'):  #'s' key
        cv2.imwrite('images/img' + str(n) + '.png', img)
        print("image nr" + str(n) + " saved")
        n += 1

vidcap.release()
cv2.destroyAllWindows()
