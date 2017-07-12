from PIL import Image
import processing

filename = '54.png'
img = Image.open(filename).convert('L')
w, h = img.size
pixels = img.load()
process = processing.Process(w, h, pixels)
img = process.make()
