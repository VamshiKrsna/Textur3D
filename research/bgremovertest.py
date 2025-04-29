from rembg import remove 
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = Image.open('naruto.jfif')

clean = remove(img)

clean = clean.convert("RGB")

plt.imshow(clean)
plt.axis('off')
plt.show()

clean.save('naruto_nobg.png')

# IT WORKS GREAT.