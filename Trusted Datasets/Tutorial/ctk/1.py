import customtkinter as ctk

# Set the theme (optional)
ctk.set_appearance_mode("Dark")  # Can be "Dark" or "Light"

# Create the main application window
app = ctk.CTk()

# Set the title of the window
app.title("CustomTkinter Example")

# Create a button widget
button = ctk.CTkButton(app, text="Click Me")
button.pack(pady=20)  # Add some padding for aesthetics

# Start the main application loop
app.mainloop()