import cv2

img = cv2.imread("Images/dog.png")
gray = cv2.imread("Images/dog.png",cv2.IMREAD_GRAYSCALE)

#cv2.imshow("Dog Image",img)
cv2.imshow("Dog Image",gray)

cv2.waitKey(0)

cv2.destroyAllWindows()