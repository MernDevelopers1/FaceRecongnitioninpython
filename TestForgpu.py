# from deepface import DeepFace
# import tensorflow as tf

# print("GPUs:", tf.config.list_physical_devices('GPU'))

# # Example verification (will run on GPU if available)
# result = DeepFace.verify(
#     img1_path="/mnt/d/TestImages/Canon EOS 1200D767.JPG",
#     img2_path="/mnt/d/TestImages/Canon EOS 1200D767.JPG",
#     model_name="Facenet512",
#     detector_backend="retinaface",
#     enforce_detection=False
# )
# print(result)
import dlib
print(dlib.DLIB_USE_CUDA)   # should print True
print(dlib.cuda.get_num_devices())  # should show your GPU count
