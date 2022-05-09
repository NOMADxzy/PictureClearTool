# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: image_procedure.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15image_procedure.proto\";\n\x08\x42\x36\x34Image\x12\x10\n\x08\x62\x36\x34image\x18\x01 \x01(\t\x12\r\n\x05width\x18\x02 \x01(\x05\x12\x0e\n\x06height\x18\x03 \x01(\x05\"+\n\nPrediction\x12\x0f\n\x07\x63hannel\x18\x04 \x01(\x05\x12\x0c\n\x04mean\x18\x05 \x01(\x02\"\x1a\n\x07\x46\x65\x61ture\x12\x0f\n\x07\x66\x65\x61ture\x18\x06 \x01(\t\"R\n\x08MatchVec\x12\x1c\n\x14known_face_encodings\x18\x07 \x01(\t\x12\x15\n\rface_encoding\x18\x08 \x01(\t\x12\x11\n\ttolerance\x18\t \x01(\x02\"M\n\x0c\x42\x36\x34Image_Box\x12\x10\n\x08\x62\x36\x34image\x18\n \x01(\t\x12\r\n\x05width\x18\x0b \x01(\x05\x12\x0e\n\x06height\x18\x0c \x01(\x05\x12\x0c\n\x04\x62oxs\x18\r \x01(\t2\xd3\x02\n\x0eImageProcedure\x12\'\n\x0bImageMeanWH\x12\t.B64Image\x1a\x0b.Prediction\"\x00\x12%\n\x0cImageFeature\x12\t.B64Image\x1a\x08.Feature\"\x00\x12!\n\x08ImagePre\x12\t.B64Image\x1a\x08.Feature\"\x00\x12!\n\x08ImageOCR\x12\t.B64Image\x1a\x08.Feature\"\x00\x12*\n\x11ImageFaceLocation\x12\t.B64Image\x1a\x08.Feature\"\x00\x12*\n\x11ImageFaceEncoding\x12\t.B64Image\x1a\x08.Feature\"\x00\x12\'\n\x0eImageFaceMatch\x12\t.MatchVec\x1a\x08.Feature\"\x00\x12*\n\x0cImageDrawBox\x12\r.B64Image_Box\x1a\t.B64Image\"\x00\x62\x06proto3')



_B64IMAGE = DESCRIPTOR.message_types_by_name['B64Image']
_PREDICTION = DESCRIPTOR.message_types_by_name['Prediction']
_FEATURE = DESCRIPTOR.message_types_by_name['Feature']
_MATCHVEC = DESCRIPTOR.message_types_by_name['MatchVec']
_B64IMAGE_BOX = DESCRIPTOR.message_types_by_name['B64Image_Box']
B64Image = _reflection.GeneratedProtocolMessageType('B64Image', (_message.Message,), {
  'DESCRIPTOR' : _B64IMAGE,
  '__module__' : 'image_procedure_pb2'
  # @@protoc_insertion_point(class_scope:B64Image)
  })
_sym_db.RegisterMessage(B64Image)

Prediction = _reflection.GeneratedProtocolMessageType('Prediction', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTION,
  '__module__' : 'image_procedure_pb2'
  # @@protoc_insertion_point(class_scope:Prediction)
  })
_sym_db.RegisterMessage(Prediction)

Feature = _reflection.GeneratedProtocolMessageType('Feature', (_message.Message,), {
  'DESCRIPTOR' : _FEATURE,
  '__module__' : 'image_procedure_pb2'
  # @@protoc_insertion_point(class_scope:Feature)
  })
_sym_db.RegisterMessage(Feature)

MatchVec = _reflection.GeneratedProtocolMessageType('MatchVec', (_message.Message,), {
  'DESCRIPTOR' : _MATCHVEC,
  '__module__' : 'image_procedure_pb2'
  # @@protoc_insertion_point(class_scope:MatchVec)
  })
_sym_db.RegisterMessage(MatchVec)

B64Image_Box = _reflection.GeneratedProtocolMessageType('B64Image_Box', (_message.Message,), {
  'DESCRIPTOR' : _B64IMAGE_BOX,
  '__module__' : 'image_procedure_pb2'
  # @@protoc_insertion_point(class_scope:B64Image_Box)
  })
_sym_db.RegisterMessage(B64Image_Box)

_IMAGEPROCEDURE = DESCRIPTOR.services_by_name['ImageProcedure']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _B64IMAGE._serialized_start=25
  _B64IMAGE._serialized_end=84
  _PREDICTION._serialized_start=86
  _PREDICTION._serialized_end=129
  _FEATURE._serialized_start=131
  _FEATURE._serialized_end=157
  _MATCHVEC._serialized_start=159
  _MATCHVEC._serialized_end=241
  _B64IMAGE_BOX._serialized_start=243
  _B64IMAGE_BOX._serialized_end=320
  _IMAGEPROCEDURE._serialized_start=323
  _IMAGEPROCEDURE._serialized_end=662
# @@protoc_insertion_point(module_scope)