import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
original_image = img.imread('dhoni.jpg')
#print(original_image.shape)
#print(original_image[0])
plt.axis('off')
plt.imshow(original_image)
img_reshaped = np.reshape(original_image,
(np.size(original_image, 0),np.size(original_image,
1)*np.size(original_image, 2)))
subplot_index = 1
for n_components in [400, 300, 200, 100, 50, 10]:
 plt.subplot(3, 2, subplot_index)
 subplot_index = subplot_index + 1
 ipca = PCA(n_components).fit(img_reshaped)
 transf_img = ipca.transform(img_reshaped)
 #restore the image from the subspace
 image_restored =
ipca.inverse_transform(transf_img)
 #reshape the image to the original array size
 image_restored = np.reshape(image_restored,
(np.size(original_image, 0),np.size(original_image,
1),np.size(original_image, 2)))
 image_restored = image_restored.astype(np.uint8)
 plt.axis('off')
 plt.title('components:' + str(n_components))
 plt.imshow(image_restored)
plt.show()