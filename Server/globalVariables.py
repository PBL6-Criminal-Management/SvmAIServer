import os

def init():
    global isTrain, isModelChanged, model_file, dataset_folder, TrainedImages_folder_id, sample_folder_id, Model_folder_id, ALLOWED_IMAGE_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_DISTANCE
    model_file = "Server/model.joblib"
    dataset_folder = "Server/TrainedImages"

    #id of TrainedImages folder: 1FcPN4UNVUZHO7JL5MPSfCgIxmhmY9LC5
    #id of Sample folder: 1PyzPF-1vIPPzWtXApsLsir8kCAt7Fpyb
    TrainedImages_folder_id = '1FcPN4UNVUZHO7JL5MPSfCgIxmhmY9LC5'
    sample_folder_id = '1PyzPF-1vIPPzWtXApsLsir8kCAt7Fpyb'
    Model_folder_id = '12oueKnaRRY4b-47WoKz04IehUhWPJnji'

    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_FILE_SIZE_MB = 50  # Set your maximum allowed file size in megabytes

    if os.path.exists(model_file):
        isTrain = True
    else:
        isTrain = False

    isModelChanged = True

def getMaxFace(faces):
    max = 0
    for face in faces:
        top, right, bottom, left = face
        area = (bottom-top) * (right-left)
        if(max < area):
            max = area
            item = (left, top, right, bottom)
    
    return item

def getIndexOfMaxFace(faces):
    max = 0
    id = -1
    for index, face in enumerate(faces):
        top, right, bottom, left = face
        area = (bottom-top) * (right-left)
        if(max < area):
            max = area
            id = index
    
    return id