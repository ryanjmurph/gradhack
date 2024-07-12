import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage

class Email:
    def __init__(self):
        self.sender_email = # your email
        self.sender_password = # watch this video to get your email password
        #https://www.youtube.com/watch?v=g_j6ILT-X0k&t=124s
        self.receiver_email = # persont to send to
        self.smtp_server = 'smtp.gmail.com'
        self.port = 465

    def send_email(self, body, image_paths=[]):
        # Create a multipart message and set headers
        em = MIMEMultipart()
        em['From'] = self.sender_email
        em['To'] = self.receiver_email
        em['Subject'] = 'Monthly Spending Report'

        # Add body to email
        em.attach(MIMEText(body, 'plain'))

        # Attach images
        for image_path in image_paths:
            with open(image_path, 'rb') as img:
                mime_image = MIMEImage(img.read())
                mime_image.add_header('Content-Disposition', f'attachment; filename="{image_path}"')
                em.attach(mime_image)

        # Create secure SSL context
        context = ssl.create_default_context()

        # Send email
        with smtplib.SMTP_SSL(self.smtp_server, self.port, context=context) as smtp:
            smtp.login(self.sender_email, self.sender_password)
            smtp.sendmail(self.sender_email, self.receiver_email, em.as_string())
