import firebase_admin
from firebase_admin import credentials, messaging

# pkmkc-sinar-firebase-adminsdk-x67kd-542ca3abd4.json
cred = credentials.Certificate("pkmkc-sinar-firebase-adminsdk-x67kd-542ca3abd4.json")
firebase_app = firebase_admin.initialize_app(cred)


def send_alert_notification(title: str, body: str, cctv_id, image_url:str=None):
    topic = "pkmkc-danger-topic"
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
            image=image_url
        ),
        topic=topic,
        data={"cctv_id":cctv_id}
    )
    messaging.send(message)
