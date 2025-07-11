🌊 Flood Detection using U-Net
This project is a flood detection tool built using a U-Net model and Streamlit. It helps identify newly flooded areas by comparing satellite images taken before and after a flood event.

The tool highlights flood zones and optionally detects if the water has turned muddy, which is a common sign of real flooding.

📦 What You Need
You will need the following files:

test.py — the main Streamlit app

unet_model.h5 — the trained U-Net model file

background.png — optional background image for the Streamlit UI

A folder named Images with test cases like:

p1b.jpg (before flood)

p1a.jpg (after flood)

Similarly up to p6a and p6b

🚀 How It Works
You upload two satellite images of the same area:

One taken before the flood

One taken after the flood

The app predicts the flooded areas in both images using the U-Net model. It then compares the masks to check if there is a significant increase in water-covered area.

If the increase is more than 5%, it says a flood is detected. If the color of the water has changed (for example, from blue to brown), it also shows an extra message indicating muddy water.

🎯 What You Get
A clear message if flood is detected or not

Optional message if the water has turned muddy

Predicted masks for both images

A red overlay showing only the newly flooded area

📁 Sample Test Cases
You can use the sample images provided in the Images folder. These include six sets of before and after images named from p1a/p1b to p6a/p6b.

💡 Notes
The images are resized to 512x512 for prediction.

The flood decision is based on area increase and optionally color shift.

You can edit test.py to adjust sensitivity or add new features.
