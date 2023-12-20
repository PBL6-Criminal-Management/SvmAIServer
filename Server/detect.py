import base64
import cv2
import numpy as np
import face_recognition
from joblib import load

from Server import globalVariables


# from matplotlib import pyplot as plt
# import matplotlib.patches as patches

model = None

# def showOnScreen(image, text_pos, text, image_pos=None):
#     # Create figure and axes
#     fig, ax = plt.subplots()

#     # Display the image
#     ax.imshow(image)

#     if image_pos is not None:
#         top, right, bottom, left = image_pos

#         # Create a Rectangle patch
#         rect = patches.Rectangle((left, top), right-left, bottom-top, linewidth=1, edgecolor='r', facecolor='none')

#         # Add the patch to the Axes
#         ax.add_patch(rect)

#     plt.text(text_pos[0], text_pos[1], text, color='white', fontsize='xx-large')

#     plt.show()

def showOnScreen2(image):
    height, width = image.shape[:2]
    scale_factor = 614/height  # You can adjust this value as needed

    resize_image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
    
    cv2.imshow("Detected Faces", resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect(image):
    global model
 
    if globalVariables.isModelChanged:
        model = load(globalVariables.model_file)
        globalVariables.isModelChanged = False

    image = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray, gray)    
    
    # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")    
    face_locations = face_recognition.face_locations(image) # model = hog by default

    max_index = globalVariables.getIndexOfMaxFace(face_locations)

    fontSize = 2*image.shape[1]/800
    color = (255, 255, 0)
    weight = 3


    if len(face_locations) >= 1:
        face_encs = face_recognition.face_encodings(image)

        if face_encs:
            labels = model.predict(face_encs)
            label = labels[max_index]

            decision = model.decision_function(face_encs)
            abs_decision = [i if i >= 0 else -i for i in decision[max_index]]
            confidence = max(abs_decision)*100/sum(abs_decision)

            print('label: ', label, 'confidence: ', confidence, '%', flush=True)                
            
            top, right, bottom, left = face_locations[max_index]
            pos = (left - 5, top - 7)

            # Only call when at localhost
            # showOnScreen(image, pos, f'Id: {labels[max_index]}', (top, right, bottom, left))

            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, f'Id: {str(label)}', pos, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, weight)

            # showOnScreen2(image)

            _, img_encoded = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

            return True, {'label': label, 'confidence': confidence}, img_base64   

    pos = (30, 30)

    # Only call when at localhost
    # showOnScreen(image, pos, 'Khong biet')

    cv2.putText(image, f'Khong biet', pos, cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, weight)

    # showOnScreen2(image)

    _, img_encoded = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

    return False, 'Không nhận diện được', img_base64


