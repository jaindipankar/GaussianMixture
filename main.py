import numpy as np
import cv2
import os
# from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

cap = cv2.VideoCapture('sky.mp4')
# Get the Default resolutions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Define the codec and filename.
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.avi', fourcc, 10, (frame_width, frame_height),0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        img2 = frame.reshape((-1, 3))
        gmm_model = GMM(n_components=2, covariance_type='full').fit(img2)  
        gmm_labels = gmm_model.predict(img2)
        original_shape = frame.shape
        segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
        shape = segmented.shape
        norm_img = np.zeros((shape[0], shape[1]))
        final_img = cv2.normalize(segmented, norm_img, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite('Segmented.jpg', final_img)
        out.write(final_img)
        # cv2.imwrite(os.path.join(path,'img' + str(cnt) + '.jpg'),final_img)
        # cnt = cnt + 1
    else:
        break

# release all the sources
cap.release()
out.release()
cv2.destroyAllWindows()