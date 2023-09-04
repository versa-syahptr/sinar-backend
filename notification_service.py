import firebase_admin
from firebase_admin import credentials, messaging
from collections import defaultdict
import time

from sinar.logger import logger

# pkmkc-sinar-firebase-adminsdk-x67kd-542ca3abd4.json
cred = credentials.Certificate("pkmkc-sinar-firebase-adminsdk-x67kd-542ca3abd4.json")
firebase_app = firebase_admin.initialize_app(cred)

last_push_time = defaultdict(int)


def send_alert_notification(title: str, body: str, cctv_id, image_url:str=None):
    if last_push_time[cctv_id] == 0 or last_push_time[cctv_id] > time.time() - 60:
        logger.info(f"send notification: {title}, {body}, {cctv_id}")
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
        last_push_time[cctv_id] = time.time()
    else:
        logger.info(f"skip notification for id: {cctv_id}")
