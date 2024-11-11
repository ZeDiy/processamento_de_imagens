import cv2
import numpy as np

image = cv2.imread('lego2.jpg')
resized_image = cv2.resize(image, (600, 400))

hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

color_ranges = [
    (([0, 100, 100], [10, 255, 255]), "Vermelho"),        
    (([10, 100, 100], [25, 255, 255]), "Laranja"),        
    (([25, 100, 100], [35, 255, 255]), "Amarelo"),        
    (([35, 100, 100], [85, 255, 255]), "Verde"),          
    (([85, 100, 100], [125, 255, 255]), "Azul"),          
    (([125, 100, 100], [160, 255, 255]), "Roxo"),         
    (([160, 100, 100], [179, 255, 255]), "Vermelho"),     
    (([8, 100, 50], [20, 150, 150]), "Marrom"),           
    (([140, 50, 100], [170, 255, 255]), "Rosa")           
]

mask = np.zeros(hsv_image.shape[:2], dtype="uint8")

masks_with_colors = []

for (lower, upper), color_name in color_ranges:
    lower_np = np.array(lower, dtype="uint8")
    upper_np = np.array(upper, dtype="uint8")
    color_mask = cv2.inRange(hsv_image, lower_np, upper_np)
    masks_with_colors.append((color_mask, color_name))
    mask = cv2.bitwise_or(mask, color_mask)

result = cv2.bitwise_and(resized_image, resized_image, mask=mask)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        
        
        piece_roi = hsv_image[y:y + h, x:x + w]
        
        
        piece_color = "unknow"
        for color_mask, color_name in masks_with_colors:
            piece_masked_area = cv2.bitwise_and(piece_roi, piece_roi, mask=color_mask[y:y + h, x:x + w])
            if cv2.countNonZero(cv2.cvtColor(piece_masked_area, cv2.COLOR_BGR2GRAY)) > 0:
                piece_color = color_name
                break
        
        
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(resized_image, f"Cor: {piece_color}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


cv2.imshow('Original', resized_image)
cv2.imshow('Mascara', mask)
cv2.imshow('Resultado', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
