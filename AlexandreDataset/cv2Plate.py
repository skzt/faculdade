import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tqdm.auto import tqdm
from random import randint

TEST_DIR = "C:/Dataset"
MODEL = "CatDog02"
ROW_SIZE = 50  # 9
COL_SIZE = 50  # 6

model = load_model(f"{MODEL}.h5")
erros = 0
acertos = 0
total = 0

# for num_folder in range(10):
# 	for img_name in os.listdir(os.path.join(TEST_DIR, str(num_folder))):
# 		img = cv2.imread(os.path.join(TEST_DIR, str(num_folder), img_name), cv2.IMREAD_GRAYSCALE)
# 		img = cv2.resize(img, (ROW_SIZE,COL_SIZE))
# 		img = img.reshape(-1, ROW_SIZE, COL_SIZE, 1)
#
# 		mdl = model.predict([img])[0]
#
# 		print("***************************")
#
# 		if mdl.argmax() == num_folder:
# 			print("Acertou")
# 			acertos+= 1
# 		else:
# 			erros+= 1
# 			print("Errou")
#
# 		total+= 1
#
# 		print(mdl.argmax())
# 		print(mdl)
# 		print(mdl[mdl.argmax()]*100)
# 		print(f"Esperado: {num_folder}")
#
# 	#input()
with open(os.path.join(TEST_DIR, f"{MODEL}2.csv"), "a") as file:
	fig = plt.figure()
	num = 0
	labels = []
	for i in range(12):
		labels.append(randint(1, 12500))

	for img_name in tqdm(labels):
		img = cv2.imread(os.path.join(TEST_DIR, "test", f"{img_name}.jpg"), cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (ROW_SIZE, COL_SIZE))

		# orig = img
		img = img.reshape(-1, ROW_SIZE, COL_SIZE, 1)

		# y = fig.add_subplot(3, 4, num+1)

		mdl = model.predict([img])
		file.write(f"[{mdl[0][0]}, {mdl[0][1]}]\n")
	# 	if np.argmax(mdl) == 1:
	# 		str_label = 'Dog'
	# 	else:
	# 		str_label = 'Cat'
	# 	print(mdl)
	# 	y.imshow(orig, cmap='gray')
	# 	plt.title(f"{str_label} {img_name}")
	# 	y.axes.get_xaxis().set_visible(False)
	# 	y.axes.get_yaxis().set_visible(False)
	# 	num += 1
	# 	if num == 12:
	# 		break
	# plt.show()


