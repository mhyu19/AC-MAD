import torch
import torch.nn.functional as F
import numpy as np
import copy
import math
from scipy.stats import entropy, wasserstein_distance
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score
import utils as utils
import model as models
import os
import argparse
import logging
import time
from termcolor import colored


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_logger(args):
    # create logger
    os.makedirs(args.output_log_dir, exist_ok=True)
    time_str = time.strftime('%m-%d-%H-%M')
    log_file = args.dataset + '_lr_' + str(args.lr) + '_norm_type_' + args.norm_type + \
               '_batch_size_' + str(args.batch_size) + '_{}.log'.format(time_str)
    final_log_file = os.path.join(args.output_log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    #
    fmt = '[%(asctime)s] %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + ' %(message)s'

    file = logging.FileHandler(filename=final_log_file, mode='a')
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console)

    return logger


class ADSMSDA():
    def __init__(self, model=models.ADS_MSDANet_TSNE(), source_loaders=0, target_loader=0, batch_size=16, iteration=2000,
                 lr=0.001, momentum=0.9, log_interval=10, id=1, save_model=None):
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval
        self.id = id
        self.save_model = save_model

    def __getModel__(self):
        return self.model

    def train(self):
        # best_model_wts = copy.deepcopy(model.state_dict())
        LEARNING_RATE = self.lr
        correct = 0
        confusion = 0
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3)
        # eta_1 = torch.nn.Parameter(torch.tensor(1.0, device=device))
        # eta_2 = torch.nn.Parameter(torch.tensor(1.0, device=device))
        # optimizer.add_param_group({"params": [eta_1, eta_2]})
        for i in range(1, self.iteration + 1):
            self.model.train()

            source_iters = [iter(self.source_loaders[i]) for i in range(len(self.source_loaders))]
            data_source = [next(source_iters[i]) for i in range(len(self.source_loaders))]
            src_data = torch.cat([data_source[i][0] for i in range(len(self.source_loaders))], axis=0)
            src_label = torch.cat([data_source[i][1] for i in range(len(self.source_loaders))], axis=0)
            s_batch_size = len(src_data)
            s_domain_label = torch.zeros(s_batch_size).long()

            target_iter = iter(self.target_loader)
            tgt_data, _ = next(target_iter)
            t_batch_size = len(tgt_data)
            t_domain_label = torch.ones(t_batch_size).long()

            src_data, src_label = src_data.to(
                device), src_label.to(device)
            tgt_data = tgt_data.to(device)

            domain_label = torch.cat([s_domain_label, t_domain_label], dim=0)
            domain_label = domain_label.to(device)
            optimizer.zero_grad()

            # get loss
            cls_loss, mmd_loss, loss_wd, lsd_loss, src_weights, tgt_weights = self.model(src_data, number_of_source=len(self.source_loaders),
                                                                data_tgt=tgt_data, label_src=src_label, mark=0)
            gamma_ = 2 / (1 + math.exp(-10 * i / self.iteration)) - 1
            beta = gamma_ / 100
            loss = cls_loss + gamma_ * (mmd_loss + loss_wd) + beta * lsd_loss
            writer.add_scalar('Loss/training cls loss', cls_loss, i)
            writer.add_scalar('Loss/training mmd loss', mmd_loss, i)
            writer.add_scalar('Loss/training lsd loss', lsd_loss, i)
            writer.add_scalar('Loss/training gamma', gamma_, i)
            writer.add_scalar('Loss/training beta', beta, i)
            writer.add_scalar('Loss/training loss', loss, i)
            writer.add_scalar('Loss/training loss_wd', loss_wd, i)
            loss.backward()
            # eta_1.data.clamp_(min=1e-3)
            # eta_2.data.clamp_(min=0.25 * eta_1.data.item())
            optimizer.step()
            t_correct, t_confusion, dict_pred = self.test(i)
            if t_correct > correct:
                correct = t_correct
                confusion = t_confusion
                if not os.path.exists('./model/' + self.save_model + '/model_csub_{}'.format(self.id[0] + 1)):
                    os.makedirs('./model/' + self.save_model + '/model_csub_{}'.format(self.id[0] + 1))
                torch.save(self.model,
                           './model/' + self.save_model + '/model_csub_{}/{}_BEST.pth'.format(self.id[0] + 1, self.id[1]))
                np.save('./model/' + self.save_model + '/model_csub_{}/{}_src_weights.npy'.format(self.id[0] + 1, self.id[1]), src_weights)
                np.save('./model/' + self.save_model + '/model_csub_{}/{}_tgt_weights.npy'.format(self.id[0] + 1, self.id[1]), tgt_weights)
                np.save('./model/' + self.save_model + '/model_csub_{}/{}_src_label.npy'.format(self.id[0] + 1, self.id[1]), src_label.cpu())
                np.save('./model/' + self.save_model + '/model_csub_{}/{}_tgt_feature.npy'.format(self.id[0] + 1, self.id[1]), dict_pred)
                # 将attention_weights中的值删除
                del src_weights
                del tgt_weights
                del dict_pred
            if i % log_interval == 0:
                logging.info(
                    'acc:{}[({:.2f}%)]\tLoss: {:.4f}\tsoft_loss: {:.4f}\tmmd_loss: {:.4f}\tloss_lsd: {:.4f}\tloss_wd: {:.4f}'.format(
                        i, t_correct, loss.item(), cls_loss.item(), mmd_loss.item(),  lsd_loss.item(), loss_wd.item(),
                    ))
        return correct, confusion

    def test(self, ti):
        self.model.eval()
        test_loss = 0
        corrects = []
        test_label = []
        pred_label = []
        raw_data = []
        pred_feature = []
        pred_feature_cfe = []
        for ti in range(len(self.source_loaders)):
            corrects.append(0)
        with torch.no_grad():
            for data_test, target in self.target_loader:
                data_test = data_test.to(device)
                target = target.to(device)
                preds, data_CFE, feature_DSFE = self.model(data_test, len(self.source_loaders))
                # data_src_CFE = torch.chunk(data_CFE, len(self.source_loaders), 0)
                for ti in range(len(preds)):
                    preds[ti] = F.softmax(preds[ti], dim=1)
                pred = sum(preds) / len(preds)
                feature_DSFE = sum(feature_DSFE) / len(feature_DSFE)
                test_loss += F.nll_loss(F.log_softmax(pred,
                                                      dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                test_label.append(target.data.squeeze().cpu())
                pred_label.append(pred.cpu())
                pred_feature.append(feature_DSFE)
                raw_data.append(data_test)
                pred_feature_cfe.append(data_CFE)
                for j in range(len(self.source_loaders)):
                    pred = preds[j].data.max(1)[1]
                    corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()
            confusion = confusion_matrix(torch.hstack(test_label), torch.hstack(pred_label))
            acc_score = accuracy_score(torch.hstack(test_label), torch.hstack(pred_label))
            pred_feature = torch.vstack(pred_feature)
            pred_feature_cfe = torch.vstack(pred_feature_cfe)
            raw_data = torch.vstack(raw_data)
            dict_pred = {'DSFE_feature': pred_feature, 'CFE_feature': pred_feature_cfe,'raw_data': raw_data,
                         'ture_label': torch.hstack(test_label), 'predict': torch.hstack(pred_label)}

            test_loss /= len(self.target_loader.dataset)
            writer.add_scalar("Test/Test loss", test_loss, ti)
        return 100. * acc_score, confusion, dict_pred


def cross_subject(data, label, session_id, subject_id, category_number, batch_size, iters, l_rate, mm,
                  log_pri, domain_weights):
    one_session_data, one_session_label = np.array(copy.deepcopy(data[session_id])), np.array(
        copy.deepcopy(label[session_id]))
    train_idxs = list(range(15))
    del train_idxs[subject_id]
    test_idx = subject_id
    tgt_data, tgt_label = copy.deepcopy(one_session_data[test_idx]), copy.deepcopy(one_session_label[test_idx])
    src_data, src_label = copy.deepcopy(one_session_data[train_idxs]), copy.deepcopy(
        one_session_label[train_idxs])

    del one_session_label
    del one_session_data

    source_loaders = []
    for j in range(len(src_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(src_data[j], src_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(tgt_data, tgt_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = ADSMSDA(model=models.ADS_MSDANet_TSNE(pretrained=False, number_of_source=len(source_loaders),
                                                  number_of_category=category_number, domain_weight=domain_weights),
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iters,
                    lr=l_rate,
                    momentum=mm,
                    log_interval=log_pri,
                    id=[session_id, subject_id],
                    save_model=model_name + dataset_name + '_cross_subject')
    # logging.info(model.__getModel__())
    acc, confusion = model.train()
    logging.info('Target_subject_id: {}, current_session_id: {}, acc: {:.2f}'.format(test_idx, session_id, acc))
    logging.info(
        'Target_subject_id: {}, current_session_id: {}, confusion: {}'.format(subject_id, session_id, confusion))
    return acc, confusion


def cross_session(data, label, session_id, subject_id, category_number, batch_size, iters, l_rate, mm,
                  log_pri, domain_weights):
    train_idxs = list(range(3))
    del train_idxs[session_id]
    test_idx = session_id

    tgt_data, tgt_label = copy.deepcopy(np.array(data[test_idx][subject_id])), copy.deepcopy(
        np.array(label[test_idx][subject_id]))

    source_loaders = []
    for j in train_idxs:
        source_loaders.append(
            torch.utils.data.DataLoader(dataset=utils.CustomDataset(data[j][subject_id], label[j][subject_id]),
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(tgt_data, tgt_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = ADSMSDA(model=models.ADS_MSDANet_TSNE(pretrained=False, number_of_source=len(source_loaders),
                                                  number_of_category=category_number, domain_weight=domain_weights),
                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iters,
                    lr=l_rate,
                    momentum=mm,
                    log_interval=log_pri,
                    id=[session_id, subject_id],
                    save_model=model_name + dataset_name + '_cross_session')
    # logging.info(model.__getModel__())
    acc, confusion = model.train()
    logging.info('Target_session_id: {}, current_subject_id: {}, acc: {}'.format(test_idx, subject_id, acc))
    logging.info(
        'Target_session_id: {}, current_subject_id: {}, confusion: {}'.format(session_id, subject_id, confusion))
    return acc, confusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AC-MSDA parameters')
    parser.add_argument('--dataset', type=str, default='seed4',
                        help='the dataset used for MS-MDAER, "seed3" or "seed4"')
    parser.add_argument('--norm_type', type=str, default='ele',
                        help='the normalization type used for data, "ele", "sample", "global" or "none"')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size for one batch, integer')
    parser.add_argument('--epoch', type=int, default=200,
                        help='training epoch, integer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--output_log_dir', default='./train/train_log_plot', type=str,
                        help='output path, subdir under output_root')
    parser.add_argument('--model_name', type=str, default='msda_plot', help='model output file')
    # python msda_adv.py --output_log_dir ./train/train_log_adv0407 --epoch 200 --batch_size 32 --model_name model_wd_msda0407
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset
    bn = args.norm_type
    logger = create_logger(args)
    # data preparation
    logging.info(f'Model name: AC-MSDA. Dataset name: {dataset_name}')
    _data, _label = utils.load_data(dataset_name)  # 3, 15,3394, 310
    logging.info(f'Normalization type: {bn}')
    if bn == 'ele':
        data_tmp = copy.deepcopy(_data)
        label_tmp = copy.deepcopy(_label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    elif bn == 'sample':
        data_tmp = copy.deepcopy(_data)
        label_tmp = copy.deepcopy(_label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminx(data_tmp[i][j])
    elif bn == 'global':
        data_tmp = copy.deepcopy(_data)
        label_tmp = copy.deepcopy(_label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.normalization(data_tmp[i][j])
    elif bn == 'none':
        data_tmp = copy.deepcopy(_data)
        label_tmp = copy.deepcopy(_label)
    else:
        pass
    trial_total, class_number, _ = utils.get_number_of_label_n_trial(
        dataset_name)

    # training settings
    BS = args.batch_size
    epoch = args.epoch
    lr = args.lr
    logging.info('BS: {}, epoch: {}'.format(BS, epoch))
    momentum = 0.9
    log_interval = 10
    iteration = 0
    if dataset_name == 'seed3':
        iteration = math.ceil(epoch * 3394 / BS)
    elif dataset_name == 'seed4':
        iteration = math.ceil(epoch * 820 / BS)
    else:
        iteration = 5000
    logging.info('Iteration: {}'.format(iteration))
    # store the results
    c_sub = []
    c_sesn = []
    cfm = []
    # iteration = 100
    # cross-validation, LOSO
    for session_id_main in range(3):
        for subject_id_main in range(15):
            js = []
            source_list = list(range(15))
            del source_list[subject_id_main]
            target_data, target_label = copy.deepcopy(data_tmp[session_id_main][subject_id_main]), copy.deepcopy(
                label_tmp[session_id_main][subject_id_main])
            for i in source_list:
                source_data, source_label = copy.deepcopy(data_tmp[session_id_main][i]), copy.deepcopy(
                    label_tmp[session_id_main][i])
                source_pdf = entropy(source_data, base=math.e)
                target_pdf = entropy(target_data, base=math.e)
                w_ds = wasserstein_distance(source_pdf, target_pdf)
                js.append(w_ds)
            gamma = 1.0
            exponents = [-gamma * j for j in js]
            max_exp = max(exponents)  # 避免数值溢出
            exp_values = [math.exp(exp - max_exp) for exp in exponents]  # 减去最大值
            # 计算softmax权重
            sum_exp = sum(exp_values)
            weights = [ev / sum_exp for ev in exp_values]
            weights = np.array(weights)
            weights = torch.tensor(weights, dtype=torch.float32)
            temp_c_sub, temp_cfm = cross_subject(data_tmp, label_tmp, session_id_main, subject_id_main,
                                                 class_number, BS, iteration, lr, momentum,
                                                 log_interval, weights)
            c_sub.append(temp_c_sub)
            cfm.append(temp_cfm)
    c_sub = np.reshape(np.array(c_sub), [3, 15])
    # c_sub = np.array(c_sub)
    logging.info(f"Cross-subject: {c_sub}")
    logging.info(f"Cross-subject mean: {np.mean(c_sub)} std: {np.std(c_sub)}, confusion: {np.sum(cfm, axis=0)}")

    # for subject_id_main in range(15):
    #     for session_id_main in range(3):
    #         js = []
    #         source_list = list(range(3))
    #         del source_list[session_id_main]
    #         target_data, target_label = copy.deepcopy(data_tmp[session_id_main][subject_id_main]), copy.deepcopy(
    #             label_tmp[session_id_main][subject_id_main])
    #         for i in source_list:
    #             source_data, source_label = copy.deepcopy(data_tmp[i][subject_id_main]), copy.deepcopy(
    #                 label_tmp[i][subject_id_main])
    #             source_pdf = entropy(source_data, base=math.e)
    #             target_pdf = entropy(target_data, base=math.e)
    #             w_ds = wasserstein_distance(source_pdf, target_pdf)
    #             # w_ds = max(0, 0.1 - w_ds)
    #             js.append(w_ds)
    #         gamma = 1.0
    #         exponents = [-gamma * j for j in js]
    #         max_exp = max(exponents)  # 避免数值溢出
    #         exp_values = [math.exp(exp - max_exp) for exp in exponents]  # 减去最大值
    #         # 计算softmax权重
    #         sum_exp = sum(exp_values)
    #         weights = [ev / sum_exp for ev in exp_values]
    #         weights = np.array(weights)
    #         # 转为tensor
    #         weights = torch.tensor(weights, dtype=torch.float32)
    #         # js = np.array(js)
    #         # weights = torch.tensor(js, dtype=torch.float32)
    #         temp_csesn, temp_cfm = cross_session(data_tmp, label_tmp, session_id_main, subject_id_main,
    #                                              class_number, BS, iteration, lr, momentum,
    #                                              log_interval, weights)
    #         c_sesn.append(temp_csesn)
    #         cfm.append(temp_cfm)
    # c_sesn = np.reshape(np.array(c_sesn), [3, 15])
    # logging.info(f"Cross-session: ACC: {c_sesn}")
    # logging.info(f"Cross-subject: ACC: {c_sub}")
    # logging.info(f"Cross-session mean:{np.mean(c_sesn)} std: {np.std(c_sesn)}, confusion: {np.sum(cfm, axis=0)}")
    # logging.info(f"Cross-subject mean: {np.mean(c_sub)} std: {np.std(c_sub)}, confusion: {np.sum(cfm, axis=0)}")
