import smtplib
from email.mime.text import MIMEText
from email.header import Header
import random


class SendEmail:
    '''
    receiver: 收件人邮箱
    send_server: 发件smtp服务器
    user: 发件邮箱
    password: 密码或授权码
    code_length: 验证码长度

    返回值：
        {
            status: '',
            code: ''
        }

    '''
    def __init__(self, receiver: str, sender_server: str, user: str, password: str, code_length: int):
        self.receiver = receiver
        self.sender_server = sender_server
        self.user = user
        self.password = password
        self.code_length = code_length

    def send(self) ->dict:
        verification_code = ''.join([str(i) for i in random.sample(range(0, 10), self.code_length)])
        msg = MIMEText('<html>您的验证码是：<h1>%s</h1>请不要泄露给他人</html>' % verification_code, 'html', 'utf-8')
        subject = '验证码'
        msg['subject'] = Header(subject, 'utf-8')

        smtp = smtplib.SMTP()
        smtp.connect(self.sender_server)
        smtp.login(self.user, password=self.password)
        noop = smtp.noop()
        smtp.sendmail(self.user, self.receiver, msg.as_string())
        smtp.quit()
        print(noop)
        if noop[0] == '250':
            response = {'status': 'ok', 'code': verification_code}
        else:
            response = {'status': 'error', 'code': ''}
        return response


if __name__ == '__main__':
    se = SendEmail(receiver='17863110068@163.com', sender_server='smtp.qq.com', user='1711074598@qq.com',
                   password='', code_length=4)
    se.send()
