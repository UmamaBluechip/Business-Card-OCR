from flask import Flask, render_template, request, redirect, url_for
from imutils.perspective import four_point_transform
import pytesseract
import imutils
import cv2
import re
import os
from werkzeug.utils import secure_filename

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image and extract information
            details, transformed_image_path = process_image(filepath)
            
            return render_template("index.html", details=details, image=transformed_image_path)
    
    return render_template("index.html")

def process_image(image_path):
    # Load the image
    orig = cv2.imread(image_path)
    image = orig.copy()
    image = imutils.resize(image, width=600)
    ratio = orig.shape[1] / float(image.shape[1])

    # Preprocess the image (grayscale, blur, edge detection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    # Find contours and get the card outline
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    cardCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            cardCnt = approx
            break
    
    if cardCnt is None:
        return {"error": "Could not detect the card outline."}, None

    # Apply perspective transform
    card = four_point_transform(orig, cardCnt.reshape(4, 2) * ratio)

    # OCR to extract text
    rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)

    # Extract phone numbers, emails, and names using regex
    phoneNums = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)
    nameExp = r"^[\w'\-,.][^0-9_!¡?÷?¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}"
    names = re.findall(nameExp, text)

    # Save the transformed card image
    transformed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "transformed_" + os.path.basename(image_path))
    cv2.imwrite(transformed_image_path, card)

    # Prepare extracted details for display
    details = {
        "phone_numbers": phoneNums,
        "emails": emails,
        "names": names
    }
    return details, transformed_image_path

if __name__ == "__main__":
    app.run(debug=True)
