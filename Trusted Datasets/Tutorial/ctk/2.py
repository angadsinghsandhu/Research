import customtkinter as ctk

# Set the theme (optional)
ctk.set_appearance_mode("Dark")  # Can be "Dark" or "Light"
# ctk.set_icon("path/to/icon.png")  # Set the icon of the window
# ctk.set_font("Arial", 12)  # Set the default font and size
# ctk.set_bg("black")  # Set the default background color
# ctk.set_fg("white")  # Set the default foreground color
ctk.set_default_color_theme("dark-blue")  # Set the default color theme

# Create the main application window
root = ctk.CTk()

# Set the title of the window
root.title("CustomTkinter Button Example")

# root.iconbitmap("path/to/icon.ico")  # Set the icon of the window
root.geometry("600x350")  # Set the size of the window

def hello():
    print("Hello, World!")

# # Create a button widget
button = ctk.CTkButton(root, text="Hello, World!", command=hello)
button.pack(pady=20)  # Add some padding for aesthetics

# Start the main application loop
root.mainloop()