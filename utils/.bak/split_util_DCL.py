#coding=utf8
from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import numpy as np
from math import ceil
from torch.nn import L1Loss
from torch import nn
import datetime

from tensorboardX import SummaryWriter


import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S") 


def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          data_ver='all',
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):
    def eval_turn(global_step, check_point_loss):
        # val phase
        model.train(False)  # Set model to evaluate mode

        val_corrects1 = 0
        val_corrects2 = 0
        val_corrects3 = 0
        val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)
        t0 = time.time()
        rec_ce_loss = 0
   
        with torch.no_grad():
            for batch_cnt_val, data_val in enumerate(data_loader['val']):
                # print data
                inputs,  labels, labels_swap, swap_law, img_name = data_val

                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
                labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).long().cuda())
                swap_law = Variable(torch.from_numpy(np.array(swap_law)).float().cuda())
                # forward
                if len(inputs)==1:
                    inputs = torch.cat((inputs,inputs))
                    labels = torch.cat((labels,labels))
                    labels_swap = torch.cat((labels_swap,labels_swap))
                outputs = model(inputs) 
                val_loss = 0 
                if isinstance(outputs, list):
                    ce_loss = criterion(outputs[0], labels)
                    rec_ce_loss += ce_loss
                    val_loss += ce_loss
                    swap_loss = criterion(outputs[1], labels_swap)
                    val_loss += swap_loss
                    law_loss = add_loss(outputs[2], swap_law)
                    val_loss += law_loss
                    outputs1 = outputs[0] + outputs[1][:,0:Config.numcls] + outputs[1][:,Config.numcls:2*Config.numcls]
                    outputs2 = outputs[0]
                    outputs3 = outputs[1][:,0:Config.numcls] + outputs[1][:,Config.numcls:2*Config.numcls
            
                print('eval batch: %d      loss : %f' % (batch_cnt_val, val_loss.item()))

                top3_val1, top3_pos1 = torch.topk(outputs1, 3)
                top3_val2, top3_pos2 = torch.topk(outputs2, 3)
                top3_val3, top3_pos3 = torch.topk(outputs3, 3)
                
                batch_corrects1 = torch.sum((top3_pos1[:, 0] == labels)).data.item()
                val_corrects1 += batch_corrects1
                batch_corrects2 = torch.sum((top3_pos1[:, 1] == labels)).data.item()
                val_corrects2 += (batch_corrects2 + batch_corrects1)
                batch_corrects3 = torch.sum((top3_pos1[:, 2] == labels)).data.item()
                val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

            # val_acc = 0.5 * val_corrects / len(data_set['val'])
            val_acc1 = 0.5 * val_corrects1 / len(data_set['val'])
            val_acc2 = 0.5 * val_corrects2 / len(data_set['val'])
            val_acc3 = 0.5 * val_corrects3 / len(data_set['val'])
        
            rec_ce_loss = rec_ce_loss.item()
            rec_ce_loss /= (test_epoch_step*test_batch_size)
            rec_checkpoint_loss = check_point_loss.item()
            rec_checkpoint_loss /= (checkpoint*test_batch_size)

        log_file.write(str(global_step) + '\t' +str(rec_checkpoint_loss)+'\t' + str(rec_ce_loss) + '\t' + str(val_acc1) + '\t' + str(val_acc3) + '\n')

        t1 = time.time()
        since = t1-t0
        print('--'*30) 
        print('current lr:%s' % exp_lr_scheduler.get_lr())
        print('%s epoch[%d]-val-loss: %.4f ||val-acc@1: %.4f val-acc@2: %.4f val-acc@3: %.4f ||time: %d' % (dt(), epoch, val_loss, val_acc1, val_acc2, val_acc3, since))

        # save model
        save_path = os.path.join(save_dir, 'weights_%d_%d_%.4f.pth'%(epoch,batch_cnt,val_acc1))
        torch.save(model.state_dict(), save_path)
        print.info('saved model to %s' % (save_path))
        print.info('--' * 30)


    step = 0
    checkpoint_list = []
    train_batch_size = data_loader['train'].batch_size
    test_batch_size = data_loader['val'].batch_size
    epoch_step = data_loader['train'].__len__()
    test_epoch_step = data_loader['val'].__len__()
    date_suffix = dt() 
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    selflog_file = open('./logs/private_log_%s.txt' % date_suffix, 'a')
    log_file = open('./logs/formal_log_r50_dcl_%s_%s_%s.log'%(data_ver, str(data_size), date_suffix), 'a')
    add_loss = L1Loss()

    train_writer = SummaryWriter('train_log')
    checkpoint_loss = 0

    for epoch in range(start_epoch,epoch_num-1):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode
        
        for batch_cnt, data in enumerate(data_loader['train']):

            step+=1
            loss = 0
            model.train(True)
            inputs, labels, labels_swap, swap_law = data
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).cuda())
            labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).cuda())
            swap_law = Variable(torch.from_numpy(np.array(swap_law)).float().cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            if isinstance(outputs, list):
                ce_loss = criterion(outputs[0], labels)
                loss += ce_loss.data[0]
                swap_loss = criterion(outputs[1], labels_swap)
                loss += swap_loss.data[0]
                law_loss = add_loss(outputs[2], swap_law)
                loss += law_loss.data[0]
                checkpoint_loss += ce_loss/train_batch_size
                selflog_file.write(dt() +'      ' + str(step) + ' : ' +str(loss)+'\n')
            print(str(step) +' : ' + str(loss.item()))
            loss.backward()
            optimizer.step()
            if step % checkpoint == 0:
                train_writer.add_scalar('ce_loss', ce_loss, step/checkpoint)
                train_writer.add_scalar('swap_loss', swap_loss, step/checkpoint)
                train_writer.add_scalar('law_loss', law_loss, step/checkpoint)
                train_writer.add_scalar('overall', loss, step/checkpoint)
                eval_turn(1.0*step/epoch_step, checkpoint_loss)  
                checkpoint_loss = 0
            elif step % savepoint == 0:
                train_writer.add_scalar('ce_loss', ce_loss, step/checkpoint)
                train_writer.add_scalar('swap_loss', swap_loss, step/checkpoint)
                train_writer.add_scalar('law_loss', law_loss, step/checkpoint)
                train_writer.add_scalar('overall', loss, step/checkpoint)
                train_writer.export_scalars_to_json("./logs/iter_scalars.json")
                save_path = os.path.join(save_dir, 'checkpoint_weights-%d-%s.pth'%(step, dt())) 
                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                torch.save(model.state_dict(), save_path)

    train_writer.export_scalars_to_json("./logs/all_scalars.json")
    train_writer.close()
    log_file.close()
    selflog_file.close()



