#coding=utf-8
import os
import json
import csv
import argparse
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets, models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from transforms import transforms
from models.LoadModel import MainModel
#from models.primitive_dcl import MainModel
from utils.save4submit import Submit_result
from dataset.dataset_DCL import collate_fn4train, collate_fn4test, collate_fn4val, dataset
from config import LoadConfig, load_data_transformers
from utils.test_tool import set_text, save_multi_img, cls_base_acc

import pdb

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='butterfly', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--b', dest='batch_size',
                        default=16, type=int)
    parser.add_argument('--nw', dest='num_workers',
                        default=16, type=int)
    parser.add_argument('--ver', dest='version',
                        default='val', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None, type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    #parser.add_argument('--swap_num', dest='swap_num',
    #                    default=7, type=int)
    parser.add_argument('--bad_case',dest='bad_case', action='store_true')
    parser.add_argument('--ss', dest='save_suffix',
                        default=None, type=str)
    parser.add_argument('--eval_analysis', dest='analysis', action='store_true')
    parser.add_argument('--submit', dest='submit',
                        action='store_true')
    parser.add_argument('--acc_report', dest='acc_report',
                        action='store_true')
    parser.add_argument('--tencroped', dest='tencroped',
                        action='store_true')
    parser.add_argument('--ensamble', dest='ensamble',
                        action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.submit:
        args.version = 'test'
        if args.save_suffix == '':
            raise Exception('**** miss --ss save suffix is needed. ')

    Config = LoadConfig(args, args.version)
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)
    data_set = dataset(Config,\
                       anno=Config.val_anno if args.version == 'val' else Config.test_anno ,\
                       unswap=transformers["None"],\
                       swap=transformers["None"],\
                       totensor=transformers["Tencrop"] if args.tencroped else transformers['test_totensor'],\
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set,\
                                             batch_size=args.batch_size,\
                                             shuffle=False,\
                                             num_workers=args.num_workers,\
                                             collate_fn=collate_fn4test)

    setattr(dataloader, 'total_item_len', len(data_set))

    save_result = Submit_result(args.dataset)
    cudnn.benchmark = True

    model = MainModel(Config)
    #resume = args.resume
    #resume = "./net_model/_5312_herb/weights_86_475_0.8466_0.9448.pth"
    #resume = "./net_model/buff_weights_3_2271_0.8835_0.9621.pth"
    #resume = "weights_7_1967_0.8765_0.9542.pth"
    #resume = './weights_12_563_0.8892_0.9581.pth'
    #resume = '/home/liang/DCL2/submitf-4.22/net_model/4-22-3_butterfly/weights-10-3999-[0.8889].pth'
    #resume = './net_model/butterfly_all_5514/_5514_butterfly/weights_17_6207_0.9334_0.9877.pth'
    #resume = './focal_weights_9_3422_0.8863_0.9560.pth'
    #resume = './net_model/buffall_51414_butterfly/weights_15_3967_0.9539_0.9846.pth'
    resume = './../re_DCL/focal_weights_9_3422_0.8863_0.9560.pth'
    model_dict=model.state_dict()
    pretrained_dict=torch.load(resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model = nn.DataParallel(model)
    criterion = CrossEntropyLoss()
    model.train(False)
    with torch.no_grad():
        val_corrects1 = 0
        val_corrects2 = 0
        val_corrects3 = 0
        val_size = ceil(len(data_set) / dataloader.batch_size)
        result_gather = {}
        acc1_bad_list = []
        acc2_bad_list = []
        acc3_bad_list = []
        gather_score = []
        gather_name = []
        count_bar = tqdm(total=dataloader.__len__())
        for batch_cnt_val, data_val in enumerate(dataloader):
            count_bar.update(1)
            inputs, labels, img_name = data_val
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
            # forward
            if args.tencroped:
                bs_, ncrops_, c_, h_, w_ = inputs.size()
                outputs = model(inputs.view(-1, c_, h_, w_))
                outputs = outputs[0].view(bs_, ncrops_, -1).mean(1)
            else:
                outputs = model(inputs)
                outputs = outputs[0]

            if args.ensamble:
                gather_score.append(F.log_softmax(outputs, dim=1))
                gather_name.extend(img_name)
                continue

            top3_val, top3_pos = torch.topk(outputs, 3)


            if args.version == 'val':
                batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
                val_corrects1 += batch_corrects1
                batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
                val_corrects2 += (batch_corrects2 + batch_corrects1)
                batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
                val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

            if args.bad_case:
                acc1_bad = (top3_pos[:, 0] != labels)
                acc1_bad_name = [[img_name[x], top3_pos[:, 0].tolist()[x], labels.tolist()[x]] for x in torch.nonzero(acc1_bad)]
                acc1_bad_list.extend(acc1_bad_name)

                acc2_bad = (top3_pos[:, 1] != labels).int()*acc1_bad.int()
                acc2_bad = (acc2_bad == 1)
                acc2_bad_name = [[img_name[x], top3_pos[:, 1].tolist()[x], labels.tolist()[x]] for x in torch.nonzero(acc2_bad)]
                acc2_bad_list.extend(acc2_bad_name)

                acc3_bad = (top3_pos[:, 2] != labels).int()*acc2_bad.int()*acc1_bad.int()
                acc3_bad = (acc3_bad == 1)
                acc3_bad_name = [[img_name[x], top3_pos[:, 2].tolist()[x], labels.tolist()[x]] for x in torch.nonzero(acc3_bad)]
                acc3_bad_list.extend(acc3_bad_name)

            if args.submit:
                if args.version != 'test':
                    raise Exception('**** should not submit validation')
                for sub_name, sub_cat in zip(img_name, top3_pos.tolist()):
                    result_gather[sub_name] = '%d %d %d'%(sub_cat[0], sub_cat[1], sub_cat[2])

            if args.analysis or args.acc_report:
                for sub_name, sub_cat, sub_val, sub_label in zip(img_name, top3_pos.tolist(), top3_val.tolist(), labels.tolist()):
                    result_gather[sub_name] = {'top1_cat': sub_cat[0], 'top2_cat': sub_cat[1], 'top3_cat': sub_cat[2],
                                               'top1_val': sub_val[0], 'top2_val': sub_val[1], 'top3_val': sub_val[2],
                                               'label': sub_label}
    if args.acc_report or args.submit or args.analysis:
        torch.save(result_gather, 'result_gather_%s'%resume.split('/')[-1][:-4]+ '.pt')

    count_bar.close()

    if args.ensamble:
        gather_score = torch.cat(gather_score, 0)
        data = {'name':gather_name,
                'score':gather_score}
        torch.save(data, 'focal_weighted_score.pt')
        raise Exception('tmp save score')

        if not os.path.exists('./ensamble'):
            os.mkdir('./ensamble')
        gather_names = np.concatenate(gather_name)
        gather_scores = np.concatenate(gather_score)
        ensamble_result = {'id':gather_names, 'probs':gather_scores}
        ensamble_io = open('./ensamble/ensamble_%s_.pkl'%resume.split('/')[-1][:-4])
        pickle.dump(ensamble_result, ensamble_io)
        ensamble_io.close()
        print('ensamble file saved ...')


    if args.bad_case:
        bad_case = {}
        #bad_case = json.load(open('bad_case.json'))
        #acc1_bad_list = bad_case['rank1_miss']
        bad_case['rank1_miss'] = acc1_bad_list
        bad_case['rank2_miss'] = acc2_bad_list
        bad_case['rank3_miss'] = acc3_bad_list
        bad_case_file = open('bad_case.json', 'w')
        json.dump(bad_case, bad_case_file)
        bad_case_file.close()
        case_img = []
        case_txt = []
        for case_item in acc1_bad_list:
            bad_name, wrong_pred, gt_label = case_item
            case_img.append(os.path.join(Config.rawdata_root, bad_name))
            case_txt.append('p:'+str(wrong_pred) + ' g:'+ str(gt_label))
        save_multi_img(case_img[:100], case_txt)

    if args.acc_report:

        val_acc1 = val_corrects1 / len(data_set)
        val_acc2 = val_corrects2 / len(data_set)
        val_acc3 = val_corrects3 / len(data_set)
        print('%sacc1 %f%s\n%sacc2 %f%s\n%sacc3 %f%s\n'%(8*'-', val_acc1, 8*'-', 8*'-', val_acc2, 8*'-', 8*'-',  val_acc3, 8*'-'))

        cls_top1, cls_top3, cls_count = cls_base_acc(result_gather)

        acc_report_io = open('acc_report_%s_%s.json'%(args.save_suffix, resume.split('/')[-1]), 'w')
        json.dump({'val_acc1':val_acc1,
                   'val_acc2':val_acc2,
                   'val_acc3':val_acc3,
                   'cls_top1':cls_top1,
                   'cls_top3':cls_top3,
                   'cls_count':cls_count}, acc_report_io)
        acc_report_io.close()




    elif args.submit:
        torch.save(result_gather, './result/%s_result_gather.pt'%args.save_suffix)
        if not save_result(result_gather, args.save_suffix):
            print('****error in write csv')
            torch.save(result_gather, './result/%s_result_gather.pt'%args.save_suffix)
            pdb.set_trace()

    if args.analysis:
        val_anno = pd.read_csv(os.path.join(Config.anno_root, 'val.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        img_names = val_anno['ImageName'].tolist()
        labels = val_anno['label'].tolist()
        anno_dict = {}
        for name, label in zip(img_names, labels): anno_dict[name] = label
        label_count = [labels.count(x) for x in range(Config.numcls)]

        cat_err = [[] for x in range(Config.numcls)]
        for nane in img_names:
            if result_gather[name]['top1_cat'] != anno_dict[name]:
                cat_err[anno_dict[name]].append(name)
        torch.save(cat_err, './result/cat_err.pt')

        cat_num = 0
        for cont in cat_err:
            if len(cont) == 0:
                continue
            else:
                error_rate = len(cont) / cat_err[cat_num]
            img_list = []
            img_counter = 0
            for img in cont:
                img_list.append(cv2.imread(os.path.join(Config.rawdata_root, img)))
                img_counter += 1
                if img_counter == 25:
                    break
            save_multi_img(img_list=img_list, save_name='cat'+str(cat_num)+'_'+str(error_rate)+'_')









