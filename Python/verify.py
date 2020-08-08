# import libraries
import requests, cv2
import json

# define init function
def init(filepath=None):
    # define global variable
    global endpoint, key, path

    # find scipt location
    if(filepath != None):
        path = filepath
    else:
        path="../"

    # Azure API endpoint
    endpoint = json.load(open(path+"config.file"))["endpoint"]

    # fetch API KEY
    key =  json.load(open(path+"config.file"))["key"]

# define run function
def getFaceID():
    # Detect Api endpoint
    api = endpoint.format("detect")

    # API argument
    params = {"returnFaceId":"true", "returnFaceAttributes": "age,gender,emotion,blur,exposure,noise"}

    # API header
    headers = {"Content-Type": "application/octet-stream", "Ocp-Apim-Subscription-Key": key}

    # create camera object
    camera = cv2.VideoCapture(0)

    # while camera is open
    while(camera.isOpened()):
        # read image from camera
        status, frame = camera.read()
        # release camera resources
        camera.release()
        # destroy all open window
        cv2.destroyAllWindows()
        # if image captured break
        if status == True:
            break

    # save image
    cv2.imwrite("Authentication-score.png",frame)

    # encode data into PNG format
    body = cv2.imencode(".png", frame)[1].tostring()

    # ping Azure API
    response = requests.post(data=body, url=api, headers=headers, params=params)

    # return response
    return(response)

# verify face using faceId
def verifyFaceID(faceId1, faceId2):
    status = dict()

    # verify Api endpoint
    api = endpoint.format("verify")

    # API header
    headers = {"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": key}

    # data
    body = {"faceId1": faceId1, "faceId2": faceId2}

    # ping Azure API
    response = requests.post(json=body, url=api, headers=headers)

    # return response
    return(response)

# execute face authentication
def run():
    # define return status dict
    status = dict()

    # fetch faceId
    faceId1 = json.load(open(path+"config.file"))["faceId"]

    # detect new faceId
    response = getFaceID()

    # update status value
    status.update(response.json()[0])

    # fetch faceId2 from response
    if(response.status_code == 200):
        # faceId2
        faceId2 = response.json()[0]["faceId"]

        # verify face
        response = verifyFaceID(faceId1, faceId2)

        if(response.status_code == 200):
            status["isIdentical"] = response.json()["isIdentical"]
            status["confidence"] = response.json()["confidence"]
            return(json.dumps(status))
        else:
            # send error message
            status["status"] = "Error: Not able to verify face"
            status["code"] = response.status_code
            return(json.dumps(status))

    else:
        # send error message
        status["status"] = "Error: Not able to get FaceId"
        status["code"] = response.status_code
        return(json.dumps(status))

if __name__ == "__main__":
    init()
    status = run()
    print("Status {}".format(status))
