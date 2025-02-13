import requests as re
from config import args, log

def send_msg():
    url = "https://www.autodl.com/api/v1/wechat/message/send"
    headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjI1MzU5LCJ1dWlkIjoiMTU4MzdiNzMtOWI3ZS00MTNhLTkxYTUtYjVkNTc4ZmJlN2IyIiwiaXNfYWRtaW4iOmZhbHNlLCJiYWNrc3RhZ2Vfcm9sZSI6IiIsImlzX3N1cGVyX2FkbWluIjpmYWxzZSwic3ViX25hbWUiOiIiLCJ0ZW5hbnQiOiJhdXRvZGwiLCJ1cGsiOiIifQ.aYjK7oljKYoTjuc_5vEzyq8B3i2HrjDfSWZFYxHhJNBuyPu5f4-xh_StpboWELfJanM8UpAumaZ3hub1aUXcUQ"}
    msg = f"model={args.model}_seed={args.seed}_{args.tuning_param_middle_path[:-1]}已完成！"

    try:
        resp = re.post(url, headers=headers, json={"name": msg})
        log.info(resp.content.decode())
        log.info_("Message has been sent to WeChat.")
    except Exception as e:
        log.error(e)
        log.error("Failed to send message to WeChat.")


if __name__ == '__main__':
    send_msg()