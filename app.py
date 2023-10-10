import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import os

class DrawingBoard:
	
	path_to_model = 'model.keras'
	alphabet=['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'
          , 'eta', 'theta', 'iota', 'kappa', 'lambda', 'miu', 'niu'
          , 'ksi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'ypsilon'
          , 'phi', 'chi', 'psi', 'omega']
	
	def __init__(self):
		if os.path.isfile(self.path_to_model):
			self.model = tf.keras.models.load_model(self.path_to_model)
		
		self.root = tk.Tk()
		self.root.title("Drawing Board")

		self.canvas = tk.Canvas(self.root, width=200, height=200, bg='white')
		self.canvas.pack()

		self.image = Image.new("L", (200, 200), color=255)
		self.draw = ImageDraw.Draw(self.image)

		self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
		self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

		self.prediction_label = tk.Label(self.root, text="Model Prediction: ")
		self.prediction_label.pack()

		self.clear_button = tk.Button(self.root, text="Clear Drawing", command=self.clear_drawing)
		self.clear_button.pack()

	def on_mouse_drag(self, event):
		x, y = event.x, event.y
		radius = 8
		self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='black')
		self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)
        
	def on_mouse_release(self, event):
		image = self.get_canvas_as_array().reshape(1,14,14,1)
		guess = self.model.predict(image)
		self.prediction_label.config(text=self.alphabet[np.argmax(guess)])
		self.root.update()

	def clear_drawing(self):
		self.canvas.delete("all")
		self.image = Image.new("L", (200, 200), color=255)
		self.draw = ImageDraw.Draw(self.image)

	def get_canvas_as_array(self):
        # Resize the canvas image to (14, 14)
		resized_image = self.image.resize((14, 14))

        # Convert the image to a NumPy array
		canvas_array = np.array(resized_image)

        # Add the third dimension to make it (14, 14, 1)
		canvas_array = np.expand_dims(canvas_array, axis=-1)

		return canvas_array

	def run(self):
		self.root.mainloop()


if __name__ == "__main__":
	drawing_board = DrawingBoard()
	drawing_board.run()