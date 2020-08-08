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
    params = {"returnFaceId":"true"}

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
    cv2.imwrite("Authentication-setup.png",frame)

    # encode data into PNG format
    body = cv2.imencode(".png", frame)[1].tostring()

    # ping Azure API
    response = requests.post(data=body, url=api, headers=headers, params=params)

    # return response
    return(response)


def run():
    # define return status dict
    status = dict()

    # get new faceId
    response = getFaceID()

    if(response.status_code == 200):
        with open(path+"config.file","r") as file:
            data = json.load(file)

        data["faceId"] = response.json()[0]["faceId"]

        with open(path+"config.file","w") as file:
            data = json.dump(data,file,indent=4)

        status["status"] = True
        return(json.dumps(status))

    else:
        status["status"] = "Error: Not able to get FaceId"
        status["code"] = response.status_code
        return(json.dumps(status))

if __name__ == "__main__":
    init()
    status = run()
    print("Status {}".format(status))
