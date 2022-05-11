import json

import cv2
import grpc

# import the generated classes
import remotes.image_procedure_pb2 as image_procedure_pb2
import remotes.image_procedure_pb2_grpc as image_procedure_pb2_grpc

# data encoding
from pathlib import Path
import numpy as np
import base64
from tools.val import RPC

# open a gRPC channel
channel = grpc.insecure_channel(RPC)

# create a stub (client)
stub = image_procedure_pb2_grpc.ImageProcedureStub(channel)


# encoding image/numpy array
def extract_feat(img_path):
    frame = cv2.imread(img_path)
    data = base64.b64encode(frame)
    image_req = image_procedure_pb2.B64Image(b64image=data, width=frame.shape[0], height=frame.shape[1])
    response = stub.ImageFeature(image_req)
    return json.loads(response.feature)


def pre_boxs(img_path):
    frame = cv2.imread(img_path)
    data = base64.b64encode(frame)
    image_req = image_procedure_pb2.B64Image(b64image=data, width=frame.shape[0], height=frame.shape[1])
    response = stub.ImagePre(image_req)
    return json.loads(response.feature)


def ocrread(img_path):
    frame = cv2.imread(img_path)
    data = base64.b64encode(frame)
    image_req = image_procedure_pb2.B64Image(b64image=data, width=frame.shape[0], height=frame.shape[1])
    response = stub.ImageOCR(image_req)
    return json.loads(response.feature)


class Face_recognition():
    def face_locations(self, img_path):
        frame = cv2.imread(img_path)
        data = base64.b64encode(frame)
        image_req = image_procedure_pb2.B64Image(b64image=data, width=frame.shape[0], height=frame.shape[1])
        response = stub.ImageFaceLocation(image_req)
        return json.loads(response.feature)

    def face_encodings(self, img_path):
        frame = cv2.imread(img_path)
        data = base64.b64encode(frame)
        image_req = image_procedure_pb2.B64Image(b64image=data, width=frame.shape[0], height=frame.shape[1])
        response = stub.ImageFaceEncoding(image_req)
        return json.loads(response.feature)

    def compare_faces(self, known_face_encodings, face_encoding, tolerance=0.5):
        req = image_procedure_pb2.MatchVec(known_face_encodings=json.dumps(known_face_encodings),
                                           face_encoding=json.dumps(face_encoding),
                                           tolerance=tolerance)
        response = stub.ImageFaceMatch(req)
        ms = json.loads(response.feature)
        matches = []
        for m in ms:
            if m==1: matches.append(True)
            else: matches.append(False)
        return matches

def draw_box(img, boxs):
    data = base64.b64encode(img)
    image_req = image_procedure_pb2.B64Image_Box(b64image=data, width=img.shape[0], height=img.shape[1],boxs=json.dumps(boxs))
    response = stub.ImageDrawBox(image_req)
    b64decoded = base64.b64decode(response.b64image)
    decompressed = b64decoded  # zlib.decompress(b64decoded)
    imgarr = np.frombuffer(decompressed, dtype=np.uint8).reshape(response.width, response.height, -1)
    return imgarr



