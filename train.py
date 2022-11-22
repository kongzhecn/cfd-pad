import torch
import torch.nn as nn
import numpy as np
import argparse
from utils.dataset import LivDetDataset
import os
from models.mobilenetv2 import feature_front, feature_back, classify
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import *
from utils.evaluate import ACE_TDR_Cal
from utils.utils import data_augment, cut_out
import logging
from utils.hard_triplet_loss import HardTripletLoss


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--root", type=str, default='./')
parser.add_argument("--train_sensor", type=str, default='D')
parser.add_argument("--test_sensor", type=str, default='D')
parser.add_argument("--t2", type=float, default=1.3)
parser.add_argument("--triplet", type=float, default=0.3)
parser.add_argument("--save", type=str, default='test', help='save path')
parser.add_argument("--criteria", type=str, default='ACC', help='evaluate matric of save model')  # TDR,ACC
parser.add_argument("--data_path", type=str, default='./data_path.txt', help='label path')
parser.add_argument("--epoch_num", type=int, default=90, help='training epoch')
parser.add_argument("--opt_num", type=int, default=4)
parser.add_argument("--batch_size_for_train", type=int, default=10)
parser.add_argument("--test_batch_num", type=int, default=256)
parser.add_argument("--batch_size_for_test", type=int, default=2)
parser.add_argument("--test_split_time", type=int, default=2)
parser.add_argument("--n_holes", type=int, default=10, help='number of holes for cut out')
parser.add_argument("--length", type=int, default=96, help='the length of holes')
parser.add_argument("--opt", type=str, default='Adam', help='optimizer')
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--save_acc_thres", type=float, default=0.05)
parser.add_argument("--lr_decay_step", type=float, default=5)
parser.add_argument("--lr_decay_gamma", type=float, default=0.8)
parser.add_argument("--channel_num", type=int, default=160, help='number of channels in the choosed feature map')
parser.add_argument("--mask_num", type=int, default=30, help='number of import channels')
args = parser.parse_args()

switch = {
    'O': 'Orcathus',
    'G': 'GreenBit',
    'D': 'DigitalPersona'
    }

t_train = switch[args.train_sensor]
t_test = switch[args.test_sensor]

logger = logging.getLogger("mainModule")
logger.setLevel(level=logging.DEBUG)
if not os.path.exists(args.root + 'log/' + args.save):
    os.makedirs(args.root + 'log/' + args.save)
    print("create log" + args.save + 'log/' + args.save)
log_path = args.root + 'log/' + args.save + '/Train_' + t_train + '_test_' + t_test + '.txt'
handler = logging.FileHandler(filename=log_path, mode='w')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)


logger.info("Training Protocol")
logger.info("Train sensor:" + t_train)
logger.info("Test sensor:" + t_test)
logger.info("Epoch Total number:{}".format(args.epoch_num))
logger.info("Train Batch Size is {:^.2f} x {:^.2f}".format(args.batch_size_for_train, args.opt_num))
logger.info("Test Batch Size is {:^.2f} x {:^.2f}".format(args.test_batch_num, args.batch_size_for_test))
logger.info("Test Split Times: {:^.2f}".format(args.test_split_time))
logger.info("Triplet rate: {:^.2f}".format(args.triplet))
logger.info("t2 rate: {:^.2f}".format(args.t2))
logger.info("Learning Parameters")
logger.info("Cut Out Holes is {:^.2f}".format(args.n_holes))
logger.info("Cut Out Holes Length is {:^.2f}".format(args.length))
logger.info("Optimizer is {}".format(args.opt))
logger.info("Learning Rate is {:^.2f}".format(args.learning_rate))
logger.info("learning rate decay step is {:^.2f}".format(args.lr_decay_step))
logger.info("learning rate decay gamma is {:^.2f}".format(args.lr_decay_gamma))
logger.info("Save Acc Threshold is {:^.2f}".format(args.save_acc_thres))
logger.info("Save path: " + args.save)
logger.info("Cross Entropy2 rate is {:^.2f}".format(args.t2))
logger.info("Mask num is {}".format(args.mask_num))


def Global_Training_init():
    train_data = LivDetDataset(args.data_path, [t_train, 'train'])
    val_data = LivDetDataset(args.data_path, [t_test, 'test'])
    test_data = LivDetDataset(args.data_path, [t_test, 'test'])
    net_front = feature_front()
    net_back = feature_back()
    net_classify = classify()

    if args.train_sensor == 'O':
        train_batch_size = 1
    else:
        train_batch_size = args.batch_size_for_train

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=4)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, pin_memory=True, num_workers=2)
    return net_front, net_back, net_classify, train_loader, val_loader, test_loader


def train_whole():
    net_front, net_back, net_classify, train_loader, val_loader, test_loader = Global_Training_init()

    net_front.cuda()
    net_back.cuda()
    net_classify.cuda()
    net_loss = nn.CrossEntropyLoss().cuda()
    triplet_loss = HardTripletLoss(margin=0.1, hardest=False).cuda()

    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net_front.parameters()), "lr": args.learning_rate},
        {"params": filter(lambda p: p.requires_grad, net_back.parameters()), "lr": args.learning_rate},
        {"params": filter(lambda p: p.requires_grad, net_classify.parameters()), "lr": args.learning_rate},
    ]

    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(optimizer_dict, lr=args.learning_rate)
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(optimizer_dict, lr=args.learning_rate, momentum=0.9)

    acc_list = []

    for i in range(5):
        acc_list.append(args.save_acc_thres)

    loss_tr = 0
    loss_te = 0
    loss_val = 0
    acc_te = 0
    acc_val = 0
    tdr_val = 0
    tdr_te = 0
    channel_gap = np.zeros(args.channel_num)

    for e in range(args.epoch_num):
        t = tqdm(train_loader)
        t.set_description("Whole Epoch [{}/{}]".format(e + 1, args.epoch_num))
        if args.train_sensor == 'O':
            init_cond = True
            train_cond = False
        opt_counter = 0
        test_counter = 0
        test_dataiter = iter(test_loader)
        val_dataiter = iter(val_loader)
        for b, (imgs, ls) in enumerate(t):
            net_front.train()
            net_back.train()
            net_classify.train()

            if args.train_sensor == 'O':
                if init_cond:
                    init_cond = False
                    train_cond = False
                    img_batches, img_deps, img_rows, img_cols = imgs.shape
                    imgs_list = []
                    l_list = []
                    counter = 0

                if counter == args.batch_size_for_train:
                    counter = 0
                    init_cond = True
                    train_cond = True
                else:
                    imgs = F.interpolate(imgs, size=[img_rows, img_cols], mode='bilinear', align_corners=False)

                    imgs = data_augment(imgs)

                    imgs = cut_out(imgs, args.n_holes, args.length)

                    for im in imgs:
                        imgs_list.append(im.numpy())
                        l_list.append(ls.numpy())
                    counter += 1
                if train_cond:
                    imgs_list = torch.tensor(imgs_list)
                    l_list = torch.tensor(l_list)
                    imgs = imgs_list.type(torch.FloatTensor).cuda()
                    l = l_list.cuda().view(-1)

                    one = torch.ones_like(l)
                    l01 = torch.where(l > 1, one, l)

                    ################## first forward ################
                    feat_front = net_front(imgs)
                    feat_back = net_back(feat_front)
                    out = net_classify(feat_back)

                   

                    ################## get important channel 3 ###############
                    stand_score = F.softmax(out, dim=1).detach()[:, 1]
                    Nf, Cf, Hf, Wf = feat_front.shape
                    feature_front_repeat = feat_front.repeat(1, Cf, 1, 1).view(Nf, Cf, Cf, Hf, Wf)
                    matrix = torch.ones((Cf, Cf)).cuda()
                    ie = torch.eye(Cf).cuda()
                    matrix = matrix - ie
                    feature_front_all = matrix.view(1, Cf, Cf, 1, 1) * feature_front_repeat
                    feature_front_all = feature_front_all.transpose(0, 1)
                    feature_front_all = feature_front_all.reshape(Nf * Cf, Cf, Hf, Wf)
                    feature_back_all = net_back(feature_front_all)
                    classify_out_all = net_classify(feature_back_all)
                    classify_out_all = classify_out_all.view(Cf, Nf, 2)
                    gap = F.softmax(classify_out_all, dim=2)[:, :, 1].detach()
                    gap = abs(gap - stand_score).sum(dim=1)
                    channel_gap = channel_gap + gap.detach().cpu().numpy()
                    mask = np.argpartition(channel_gap, -1 * args.mask_num)[-1 * args.mask_num:]


                    ################# put 0 to unimportant channel ################
                    feature_front_second = net_front(imgs)
                    matrix = torch.zeros(feature_front_second.shape, device=feature_front_second.device)
                    matrix[:, mask, :, :] = 1
                    feature_front_mask = feature_front_second * matrix
                    feature_back_mask = net_back(feature_front_mask)
                    classify_out_mask = net_classify(feature_back_mask)

                    ######### cross-entropy loss #########
                    loss_cross = net_loss(out, l01)

                    ############  triplet loss  ##############
                    loss_triplet = triplet_loss(feature_back_mask, l)

                    ###########  cross-entropy loss2 ############
                    loss_cross2 = net_loss(classify_out_mask, l01)


                    crite = loss_triplet * args.triplet + loss_cross + loss_cross2 * args.t2
                    # crite = loss_cross
                    loss = crite.cpu().detach().data.numpy()
                    loss_tr = 0.6 * loss_tr + 0.4 * loss
                    crite.backward()
                    opt_counter += 1


            else:
                imgs = data_augment(imgs)

                imgs = cut_out(imgs, args.n_holes, args.length)
                imgs = imgs.type(torch.FloatTensor).cuda()
                l = ls.cuda().view(-1)

                one = torch.ones_like(l)
                l01 = torch.where(l > 1, one, l)

                feat_front = net_front(imgs)
                feat_back = net_back(feat_front)
                out = net_classify(feat_back)

                
                ################## get important channel 3 ###############
                stand_score = F.softmax(out, dim=1).detach()[:, 1]
                Nf, Cf, Hf, Wf = feat_front.shape
                feature_front_repeat = feat_front.repeat(1, Cf, 1, 1).view(Nf, Cf, Cf, Hf, Wf)
                matrix = torch.ones((Cf, Cf)).cuda()
                ie = torch.eye(Cf).cuda()
                matrix = matrix - ie
                feature_front_all = matrix.view(1, Cf, Cf, 1, 1) * feature_front_repeat
                feature_front_all = feature_front_all.transpose(0, 1)
                feature_front_all = feature_front_all.reshape(Nf * Cf, Cf, Hf, Wf)
                feature_back_all = net_back(feature_front_all)
                classify_out_all = net_classify(feature_back_all)
                classify_out_all = classify_out_all.view(Cf, Nf, 2)
                gap = F.softmax(classify_out_all, dim=2)[:, :, 1].detach()
                gap = abs(gap - stand_score).sum(dim=1)
                channel_gap = channel_gap + gap.detach().cpu().numpy()
                mask = np.argpartition(channel_gap, -1 * args.mask_num)[-1 * args.mask_num:]


                feature_front_second = net_front(imgs)
                matrix = torch.zeros(feature_front_second.shape, device=feature_front_second.device)
                matrix[:, mask, :, :] = 1
                feature_front_mask = feature_front_second * matrix
                feature_back_mask = net_back(feature_front_mask)
                classify_out_mask = net_classify(feature_back_mask)

                ######### cross-entropy loss #########
                loss_cross = net_loss(out, l01)

                ############  triplet loss  ##############
                loss_triplet = triplet_loss(feature_back_mask, l)

                # ###########  cross-entropy loss2 ############
                loss_cross2 = net_loss(classify_out_mask, l01)

                crite = loss_triplet * args.triplet + loss_cross + loss_cross2 * args.t2
                # crite = loss_cross
                loss = crite.cpu().detach().data.numpy()
                loss_tr = 0.6 * loss_tr + 0.4 * loss
                crite.backward()
                opt_counter += 1

            if opt_counter == args.opt_num:  # optimize
                optimizer.step()
                optimizer.zero_grad()
                opt_counter = 0
                test_counter += 1

            if test_counter == args.test_split_time:
                net_front.eval()
                net_back.eval()
                net_classify.eval()
                with torch.no_grad():
                    for ind_ in range(2):
                        correct = torch.zeros(1).squeeze().cuda()
                        total = torch.zeros(1).squeeze().cuda()
                        loss = torch.zeros(1).squeeze().cuda()
                        result = []
                        for j in range(args.test_batch_num):
                            for b_i in range(args.batch_size_for_test):
                                if ind_ == 0:
                                    try:
                                        img, l = test_dataiter.next()
                                    except StopIteration:
                                        test_dataiter = iter(test_loader)
                                        img, l = test_dataiter.next()
                                else:  # ind=1
                                    try:
                                        img, l = val_dataiter.next()
                                    except StopIteration:
                                        val_dataiter = iter(val_loader)
                                        img, l = val_dataiter.next()
                                img = img.type(torch.FloatTensor).cuda()
                                l = l.cuda().view(-1)

                                one = torch.ones_like(l)
                                l01 = torch.where(l > 1, one, l)

                                feat_front = net_front(img)
                                feat_back = net_back(feat_front)
                                out = net_classify(feat_back)

                                out_f = F.softmax(out, dim=1)
                                loss += net_loss(out, l01)
                                pred = torch.argmax(out_f, 1)
                                correct += (pred == l01).sum().float()
                                total += len(l01)
                                result.append(
                                    [l01.cpu().detach().data.numpy()[0], out_f.cpu().detach().data.numpy()[0, 1]])
                        if ind_ == 0:
                            acc_te = (correct / total).cpu().detach().data.numpy()
                            loss_te = (loss / total).cpu().detach().data.numpy()
                            ace, tdr_te = ACE_TDR_Cal(result)
                        else:
                            acc_val = (correct / total).cpu().detach().data.numpy()
                            loss_val = (loss / total).cpu().detach().data.numpy()
                            ace, tdr_val = ACE_TDR_Cal(result)

                test_counter = 0
                if args.criteria == 'TDR':
                    crit_save = tdr_te
                if args.criteria == 'ACC':
                    crit_save = acc_te
                if crit_save >= min(acc_list):
                    acc_list[acc_list.index(min(acc_list))] = crit_save
                    acc_list = sorted(acc_list, reverse=True)
                    if not os.path.exists(args.root + 'model/' + args.save):
                        os.makedirs(args.root + 'model/' + args.save)
                        print("create model" + args.root + 'model/' + args.save)
                    net_path_feature_front = os.path.join(args.root,
                                                          'model/' + args.save + '/Train_' + t_train + '_test_' + t_test + '_TOP_' + str(
                                                              acc_list.index(crit_save) + 1) + '_featurefront_Net.pth')
                    net_path_feature_back = os.path.join(args.root,
                                                         'model/' + args.save + '/Train_' + t_train + '_test_' + t_test + '_TOP_' + str(
                                                             acc_list.index(crit_save) + 1) + '_featureback_Net.pth')
                    net_path_classify = os.path.join(args.root,
                                                     'model/' + args.save + '/Train_' + t_train + '_test_' + t_test + '_TOP_' + str(
                                                         acc_list.index(crit_save) + 1) + '_classify_Net.pth')
                    torch.save(net_front.state_dict(), net_path_feature_front)
                    torch.save(net_back.state_dict(), net_path_feature_back)
                    torch.save(net_classify.state_dict(), net_path_classify)
                    print(
                        "Save model, Top: {} ! Criterion: {:4f}".format(str(acc_list.index(crit_save) + 1), crit_save))

            if test_counter == 0:
                t.set_postfix_str(
                    'Val_Acc:{:^2f}, Test_Acc:{:^2f}, TDR_val:{:^2f}, TDR_test:{:^2f}'.format(acc_val, acc_te, tdr_val,
                                                                                              tdr_te))
            else:
                t.set_postfix_str(
                    'TrLoss : {:^2f}, Val_Loss:{:^2f}, Test_Loss:{:^2f}'.format(loss_tr, loss_val, loss_te))
            t.update()
    for i in range(1, 6):
        ace, tdr = test_for_whole(search=[switch[args.test_sensor], 'test'], sensor_te=switch[args.test_sensor], num=i)
        logger.info("TOP%d:\n \t\t\tace:%.5f tdr:%.5f" % (i, ace, tdr))

def test_for_whole(search, sensor_te, num):
    net_front = feature_front()
    net_front.load_state_dict(torch.load(args.root + 'model/' + args.save + '/Train_' + t_train + '_test_' + t_test + '_TOP_' + str(num) + '_featurefront_Net.pth'))
    net_back = feature_back()
    net_back.load_state_dict(torch.load(args.root + 'model/' + args.save + '/Train_' + t_train + '_test_' + t_test + '_TOP_' + str(num) + '_featureback_Net.pth'))
    net_classify = classify()
    net_classify.load_state_dict(torch.load(args.root + 'model/' + args.save + '/Train_' + t_train + '_test_' + t_test + '_TOP_' + str(num) + '_classify_Net.pth'))

    test_data = LivDetDataset(args.data_path, search)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_dataiter = iter(test_loader)

    net_front.cuda()
    net_back.cuda()
    net_classify.cuda()
    net_front.eval()
    net_back.eval()
    net_classify.eval()

    correct = torch.zeros(1).squeeze().cuda()
    result = []
    with torch.no_grad():
        with tqdm(total=len(test_data.imgs), ncols=110, desc='Train:Global_log' + ' Test:' + sensor_te) as t:
            inum = 1
            for img, l in test_dataiter:
                img = img.type(torch.FloatTensor).cuda()
                l = l.cuda().view(-1)

                one = torch.ones_like(l)
                l01 = torch.where(l > 1, one, l)

                feat_front = net_front(img)
                feat_back = net_back(feat_front)
                out = net_classify(feat_back)

                out = F.softmax(out, dim=1)
                pred = torch.argmax(out, 1)
                correct += (pred == l01).sum().float()
                acc = (correct / inum).cpu().detach().data.numpy()
                result.append([l01.cpu().detach().data.numpy()[0], out.cpu().detach().data.numpy()[0, 1]])
                inum += 1

                t.set_postfix_str('ACC={:^7.3f}'.format(acc))
                t.update()

    ace, tdr = ACE_TDR_Cal(result, rate=0.01)
    print('ACE : {:^4f}    TDR@FDR=1% : {:^4f}'.format(ace, tdr))
    return ace, tdr

# def eval(train_loader, num):
#     net_front = feature_front()
#     model_dict = net_front.state_dict()
#     path = args.root + 'model/' + args.save + '/Train_' + t_train + '_test_' + t_test + '_TOP_' + str(num) + '_featurefront_Net.pth'
#     pretrained_dict = torch.load(path)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     net_front.load_state_dict(model_dict)
#     net_front.cuda()
#
#     net_back = feature_back()
#     model_dict = net_back.state_dict()
#     path = args.root + 'model/' + args.save + '/Train_' + t_train + '_test_' + t_test + '_TOP_' + str(num) + '_featureback_Net.pth'
#     pretrained_dict = torch.load(path)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     net_back.load_state_dict(model_dict)
#     net_back.cuda()
#
#     net_classify = classify()
#     model_dict = net_classify.state_dict()
#     path = args.root + 'model/' + args.save + '/Train_' + t_train + '_test_' + t_test + '_TOP_' + str(num) + '_classify_Net.pth'
#     pretrained_dict = torch.load(path)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     net_classify.load_state_dict(model_dict)
#     net_classify.cuda()
#
#     net_front.eval()
#     net_back.eval()
#     net_classify.eval()
#     with torch.no_grad():
#         correct = torch.zeros(1).squeeze().cuda()
#         total = torch.zeros(1).squeeze().cuda()
#         loss = torch.zeros(1).squeeze().cuda()
#         result = []
#         for iter, (img, l) in enumerate(train_loader):
#             img = img.type(torch.FloatTensor).cuda()
#             l = l.cuda().view(-1)
#
#             one = torch.ones_like(l)
#             l01 = torch.where(l > 1, one, l)
#
#             output_front = net_front(img)
#             output_back = net_back(output_front)
#
#             out = net_classify(output_back)
#
#             out_f = F.softmax(out, dim=1)
#
#             pred = torch.argmax(out_f, 1)
#             correct += (pred == l01).sum().float()
#             total += len(l01)
#             result.append(
#                 [l01.cpu().detach().data.numpy()[0], out_f.cpu().detach().data.numpy()[0, 1]])
#         acc_te = (correct / total).cpu().detach().data.numpy()
#         loss_te = (loss / total).cpu().detach().data.numpy()
#         ace, tdr_te = ACE_TDR_Cal(result)
#     crit_save = acc_te
#     return crit_save, tdr_te, ace

if __name__ == '__main__':
    train_whole()