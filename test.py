import pyperclip

# Define the text you want to copy to the clipboard
text_to_copy = "a\tb\tc"

# Copy the text to the clipboard
pyperclip.copy(text_to_copy)

# Notify the user
print(
    "The text has been copied to the clipboard. You can now paste it into Google Docs."
)
