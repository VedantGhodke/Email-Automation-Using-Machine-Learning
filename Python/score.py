# import libraries
import json, os, pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# define init function
def init(filepath=None):
    # define global variable
    global vocab, model, trainedVectorizer, transformer, path

    # find scipt location
    if(filepath != None):
        path = filepath + "Python\\"
    else:
        path="./"

    # load machine learning model and meta data
    vocab =  pickle.load(open(path + "TfidfVectorizerModel.pkl", "rb"))
    model = joblib.load(open(path + "MultinomialNBModel.pkl","rb"))

    # init stages using meta data
    trainedVectorizer = CountVectorizer(decode_error="replace",vocabulary=vocab)
    transformer = TfidfTransformer()

# define run function to execute the ml model
def run(raw_data):

    # define return status dict
    status = dict()

    # transform feature to feature vector
    featureVector = trainedVectorizer.fit_transform([raw_data])
    featureVector_fit = transformer.fit_transform(featureVector)

    # save result into y_hat
    status["prediction"]  = model.predict(featureVector_fit).astype(dtype=float)[0]
    status["confidence"] = max(model.predict_proba(featureVector_fit).astype(dtype=float).tolist()[0])

    # return JSON
    return(json.dumps(status))


if __name__ == "__main__":
    init()
    test = "This is to check machine learning model"
    result = run(test)
    print("Data: {}\nResult: {}".format(test,result))
