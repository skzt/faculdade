import numpy as np
import cv2
import os
from keras.models import load_model

TEST_DIR = "realImages/Teste"

model = load_model("Teste02.h5")
erros = 0
acertos = 0
total = 0

for num_folder in range(10):
	for img_name in os.listdir(os.path.join(TEST_DIR, str(num_folder))):
		img = cv2.imread(os.path.join(TEST_DIR, str(num_folder), img_name), cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (9,6))
		img = img.reshape(-1, 9, 6, 1)

		mdl = model.predict([img])[0]

		print("***************************")
		
		if mdl.argmax() == num_folder:
			print("Acertou")
			acertos+= 1
		else:
			erros+= 1
			print("Errou")
		
		total+= 1

		print(mdl.argmax())
		print(mdl[mdl.argmax()]*100)
		print(f"Esperado: {num_folder}")

	#input()

print(f"Total de números testados: {total}")
print(f"Total de acertos: {acertos}")
print(f"Total de erros: {erros}")
print(f"Porcentagem de erros: {(erros*100)/total}%")
print(f"Porcentagem de acertos: {(acertos*100)/total}%")

