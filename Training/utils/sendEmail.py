from smtplib import SMTP
import smtplib


def let_me_know(content, subject="Result", to='1679882386@qq.com'):
    HOST = "smtp.qq.com"  # 定义smtp主机
    SUBJECT = subject  # 定义邮件主题
    TO = to  # 定义邮件收件人
    FROM = "1303635317@qq.com"  # 定义邮件发件人
    text = content  # 邮件内容,编码为ASCII范围内的字符或字节字符串，所以不能写中文
    BODY = '\r\n'.join((  # 组合sendmail方法的邮件主体内容，各段以"\r\n"进行分离
        "From: %s" % "admin",
        "TO: %s" % TO,
        "subject: %s" % SUBJECT,
        "",
        text
    ))
    # server = SMTP()  # 创建一个smtp对象
    # server.connect(HOST, '465')  # 链接smtp主机
    server = smtplib.SMTP_SSL(HOST, 465)
    server.login(FROM, "xomtghcamowrhbhj")  # 邮箱账号登陆
    server.sendmail(FROM, TO, BODY)  # 发送邮件
    server.quit()  # 端口smtp链接


if __name__ == "__main__":
    let_me_know(str([2, 3, 4, 5]))
