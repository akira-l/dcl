import os
import argparse

import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

def send_mail(img_list):
    print('send email ...')
    msg = MIMEMultipart()
    msg['Subject'] = 'exp_figures'
    msg['From'] = 'xxx@outlook.com'
    msg['To'] = 'xxx@outlook.com'

    text = MIMEText("exp_result")
    msg.attach(text)

    for img in img_list:
        img_data = open(img, 'rb').read()
        image = MIMEImage(img_data, name=os.path.basename(img))
        msg.attach(image)

    s = smtplib.SMTP('smtp.outlook.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login("xxx@outlook.com", "****")
    s.sendmail(msg['From'], msg['To'], msg.as_string())
    s.quit()
    print('send email done ...')



def parse_args():
    parser = argparse.ArgumentParser(description='save para')
    parser.add_argument('--dir', dest='dir', default='./grep_log', type=str)
    args = parser.parse_args()
    return args

class save_figures(object):
    def __init__(self, folder):
        self.saved_imgs = []
        self.folder=folder
        self.data_step = 5000

    def __call__(self, data_list, name):
        data_list = [float(x) for x in data_list]
        if len(data_list) == 0:
            pdb.set_trace()
        if len(data_list) > self.data_step:
            sampled_step = len(data_list) // self.data_step
            sampled = [sum(data_list[x*sampled_step:(x+1)*sampled_step])/sampled_step for x in range(self.data_step)]
            self.plot_curve(sampled, 'sampled_'+name)
            self.plot_curve(data_list[-5000:], 'latest_'+name)
        else:
            self.plot_curve(data_list, name)

    def plot_curve(self, data_list, name):
        plt.plot(data_list)
        img_path = os.path.join(self.folder, name+'.png')
        plt.title(name)
        plt.savefig(img_path)
        plt.close()
        self.saved_imgs.append(img_path)

    def get_imgs(self):
        return self.saved_imgs


if __name__ == '__main__':
    args = parse_args()
    folder = args.dir
    exp_plot = save_figures(folder)
    loss_io = open(os.path.join(folder, 'loss'))
    loss_data = loss_io.readlines()
    trainacc_io = open(os.path.join(folder, 'train_acc'))
    trainacc_data = trainacc_io.readlines()
    valacc_io = open(os.path.join(folder, 'val_acc'))
    valacc_data = valacc_io.readlines()

    print('save loss ...')
    loss_num = len(loss_data[0].split(' '))
    print('loss gathering ...')
    loss_gather = [x.split(' ') for x in loss_data]
    for loss_item in range(loss_num):
        print('loss %d' %loss_item)
        exp_plot([x[loss_item] for x in loss_gather], 'loss'+str(loss_item))

    print('save train acc ...')
    trainacc_gather = [x.split(' ') for x in trainacc_data]
    exp_plot([x[0] for x in trainacc_gather], 'trainval_loss')
    for acc_item in range(3):
        exp_plot([x[1+acc_item] for x in trainacc_gather], 'trainacc'+str(1+acc_item))

    print('save val acc ...')
    valacc_gather = [x.split(' ') for x in valacc_data]
    exp_plot([x[0] for x in valacc_gather], 'val_loss')
    for acc_item in range(3):
        exp_plot([x[1+acc_item] for x in valacc_gather], 'valacc'+str(1+acc_item))

    img_list = exp_plot.get_imgs()
    send_mail(img_list)






