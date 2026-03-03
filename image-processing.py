import cv2
import uuid
import os
import time
import subprocess

labels = ["thumbsup", "thumbsdown", "thankyou", "livelong"]
number_imgs = 5

IMAGES_PATH = os.path.join("../Tensorflow", "workspace", "images", "collectedimages")
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path)


# for label in labels:
#     cap = cv2.VideoCapture(0)
#     print('Collecting images for {}'.format(label))
#     time.sleep(5)
#     for imgnum in range(number_imgs):
#         print("Collecting image {}".format(imgnum))
#         ret, frame = cap.read()
#         imgname = os.path.join(IMAGES_PATH, label, label + "," + "{}.jpg".format(str(uuid.uuid1())))
#         cv2.imwrite(imgname, frame)
#         cv2.imshow("frame", frame)
#         time.sleep(2)
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
# cap.release()
# cv2.destroyAllWindows()
#

LABELIMG_PATH = os.path.join("../Tensorflow", "labelimg")

# Create LabelImg directory
if not os.path.exists(LABELIMG_PATH):
    os.makedirs(LABELIMG_PATH)

    # Download LabelImg using Git
    subprocess.run(["git", "clone", "https://github.com/tzutalin/labelImg", LABELIMG_PATH])

# Operations based on operating system type
if os.name == "posix":  # Linux/macOS
    subprocess.run(["make", "qt5py3"], cwd=LABELIMG_PATH)

if os.name == "nt":  # Windows
    subprocess.run(["cd", LABELIMG_PATH, "&&", "pyrcc", "-o", "libs/resources.py", "resource.qrc"], shell=True)

subprocess.run(f"cd {LABELIMG_PATH} && python labelImg.py", shell=True)

TRAIN_PATH = os.path.join("../Tensorflow", "workspace", "images", "train")
TEST_PATH = os.path.join("../Tensorflow", "workspace", "images", "test")
ARCHIVE_PATH = os.path.join("../Tensorflow", "workspace", "images", "archive.tar.gz")

#!tar -czf {ARCHIVE_PATH} {TRAIN_PATH} {TEST_PATH}
