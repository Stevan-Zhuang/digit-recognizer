from tkinter import *
import torch

class Application:
    PIXEL_SIZE = 16
    IMAGE_SIZE = 28

    def __init__(self, model):
        self.model = model

        self.win = Tk()
        self.win.title("Digit Recognizer")
        win_size = Application.IMAGE_SIZE * Application.PIXEL_SIZE
        self.win.geometry(f"{win_size}x{win_size + 125}")
        self.font = ("Arial", 25)

        self.erase = False
        self.draw_button = Button(
            self.win, text="draw", font=self.font, command=self.set_draw
        )
        self.erase_button = Button(
            self.win, text="erase", font=self.font, command=self.set_erase
        )
        self.draw_button.grid(row=0, column=0, sticky='WE')
        self.erase_button.grid(row=0, column=1, sticky='WE')

        self.canvas = Canvas(
            self.win, bg="white", height=f"{win_size}", width=f"{win_size}"
        )
        self.canvas_data = torch.full(
            (1, 1, Application.IMAGE_SIZE, Application.IMAGE_SIZE), -1.0
        )

        self.canvas.bind("<Button-1>", self.make_dot)
        self.canvas.bind("<B1-Motion>", self.make_dot)
        self.canvas.grid(row=1, columnspan=2)

        self.label = Label(self.win, text="Draw a digit to get started!",
                           font=self.font)
        self.label.grid(row=2, columnspan=2)

        self.win.mainloop()

    def set_draw(self):
        self.erase = False

    def set_erase(self):
        self.erase = True

    def make_dot(self, event):
        fill_val = -1.0 if self.erase else 1.0
        fill_col = "white" if self.erase else "black"

        pixel_x = event.x // Application.PIXEL_SIZE
        pixel_y = event.y // Application.PIXEL_SIZE
        if (0 <= pixel_x < Application.IMAGE_SIZE and
            0 <= pixel_y < Application.IMAGE_SIZE):
            self.canvas_data[0, 0, pixel_y - 1: pixel_y + 1,
                                   pixel_x - 1: pixel_x + 2] = fill_val
            x = Application.PIXEL_SIZE * pixel_x
            y = Application.PIXEL_SIZE * pixel_y
            event.widget.create_rectangle(
                x - Application.PIXEL_SIZE, y - Application.PIXEL_SIZE,
                x + Application.PIXEL_SIZE * 2, y + Application.PIXEL_SIZE * 2,
                fill=fill_col, width=0
            )
            pred = self.model(self.canvas_data)
            self.label['text'] = f"Model guess: {pred.argmax(1).item()}"

def create_app(model):
    return Application(model)

