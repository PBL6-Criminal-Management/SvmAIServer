import os
import cv2
import numpy as np
from PIL import Image
import face_recognition

from joblib import dump

from sklearn.svm import SVC

from Server.download_file import download_folder, upload_model

from Server import globalVariables

def load_images_and_labels_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        dir_path = os.path.join(folder,label)

        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                img = np.array(Image.open(img_path))
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if img is not None:
                    images.append(img)
                    labels.append(label)

    return images, labels

def create_model():
    model = cv2.face.LBPHFaceRecognizer_create()
    return model

def prepare_data(images, labels):
    face_encs = []

    count = 0
    for index, image in enumerate(images):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray, gray)

        # face_images = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")
        face_images = face_recognition.face_locations(image) # model = hog by default

        if len(face_images) >= 1:
            face_enc = face_recognition.face_encodings(image)

            if face_enc:
                max_index = globalVariables.getIndexOfMaxFace(face_images)

                face_enc = face_enc[max_index]   # Only get 1 biggest face representative for current label from image to train

                face_encs.append(face_enc)
                print('Oke', flush=True)
                continue
        
        del labels[index - count]
        count += 1        
        print('Bo qua', flush=True)
        continue

    return np.asarray(face_encs), np.asarray(labels, dtype=np.int32)

def train_and_save_model(images, labels, output_model_file):
    faces, labels = prepare_data(images, labels)

    # Create an SVM model
    svm_model = SVC(kernel='linear', gamma='scale')

    # Train the SVM model
    svm_model.fit(faces, labels)

    os.makedirs(os.path.dirname(output_model_file), exist_ok=True)

    with open(output_model_file, 'wb') as f:
        dump(svm_model, f)

    upload_model(output_model_file, globalVariables.Model_folder_id)

    return len(labels)

def retrain():
    # not save files into system
    images, labels = download_folder(globalVariables.TrainedImages_folder_id, globalVariables.dataset_folder)

    # save files (Consumes a lot of memory)
    # count = download_folder(globalVariables.sample_folder_id, globalVariables.dataset_folder, True)
    # print(f'Download file counts: {count}', flush=True)
    # if not os.path.exists(globalVariables.dataset_folder):
    #     return 0
    # images, labels = load_images_and_labels_from_folder(globalVariables.dataset_folder)

    if len(labels) > 0:
        count = train_and_save_model(images, labels, globalVariables.model_file)
        globalVariables.isTrain = True
        globalVariables.isModelChanged = True
        return count    
    else:
        return 0

