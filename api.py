from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Initialize FastAPI app
app = FastAPI()

# Load your dataset
df = pd.read_csv("distfinaldataset.txt", sep=',', header=0)

# Define input schema using Pydantic BaseModel
class InputData(BaseModel):
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    G: float
    H: float
    I: float
    J: float
    K: float

# Initialize encoder
encoder = LabelEncoder()
df['LABEL'] = encoder.fit_transform(df['LABEL'].values)

# Define model pipeline
model = Pipeline([
    ('SVC', SVC(random_state=20))
])
model.fit(df.drop(["LABEL"], axis=1), df['LABEL'])

# Email configuration
EMAIL_ADDRESS = 'hilsavani1143@gmail.com'
EMAIL_PASSWORD = 'mwnj bwdc zida iqfj'

# Define function to send email
def send_email(receiver_email, subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

# Define prediction endpoint
@app.post("/predict/")
async def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame(data.dict(), index=[0])
    # Make prediction
    pred = model.predict(input_df)
    # Inverse transform predicted label
    pred_label = encoder.inverse_transform(pred)[0]
    
    # Send email with prediction
    receiver_email = 'vaibhavmehsana123@gmail.com'
    subject = 'Attack Detected'
    body = f"The predicted label is: {pred_label}"
    if pred_label == "attack"  :
        send_email(receiver_email, subject, body)
    
    return {"prediction": pred_label}

# Additional endpoints can be added as needed

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
