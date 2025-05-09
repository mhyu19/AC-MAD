# For SEED data loading
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import pickle
import copy
import os
import scipy.io as scio

# standard package
import numpy as np
import random

random.seed(0)
dataset_path = {'seed4': 'E:\\EEG\\datasets\\SEED_IV\\eeg_feature_smooth',
                'seed3': 'E:\\EEG\\datasets\\SEe\\ExtractedFeatures',
                'deap': 'E:\\EEG\\datasets\\DEAP\\data_preprocessed_python\\'}

'''
Tools
'''
import _pickle as cPickle
from scipy.signal import stft

fs = 128


def label_process(labels):
    """
    打标签
    :param labels: 标签
    :return: 处理后的标签
    """
    return torch.tensor(np.where(labels < 5, 0, 1), dtype=torch.long)  # 小于 5 的元素改为 0，大于等于 5 的改为 1


def extract_feature(data, fs=128):
    """
    提取特征
    :param np.ndarray data: 分割后的数据 每个受试者：[240, 32, 1280]
    :param fs: 采样频率
    :return: 特征
    """
    # 短时傅里叶变换
    f, t, zxx = stft(data, fs=fs, window='hann', nperseg=128, noverlap=0, nfft=256)
    # f, t 的长度与数据最后一位有关
    power = np.power(np.abs(zxx), 2)
    fStart = [1, 4, 8, 14, 31]  # 起始频率
    fEnd = [4, 8, 14, 31, 50]  # 终止频率
    # 计算特征
    de_time = []
    for i in range(1, len(t)):
        bands = []
        for j in range(len(fStart)):
            index1 = np.where(f == fStart[j])[0][0]
            index2 = np.where(f == fEnd[j])[0][0]
            psd = np.sum(power[:, :, index1:index2, i], axis=2) / (fEnd[j] - fStart[j] + 1)
            de = np.log2(psd)
            bands.append(de)
        de_bands = np.stack(bands, axis=-1)
        de_time.append(de_bands)
    de_f = np.stack(de_time)
    de_f = de_f[0]
    return de_f


def data_divided(raw_data, label):
    window_size = 1  # 6s时间窗口
    step = 1  # 3s步长 有重叠
    num = (60 - window_size) // step + 1  # 分割段数
    # 校准数据，前3秒数据分割
    baseline_time = 3
    _, real_data = np.split(raw_data, [baseline_time * fs], axis=-1)
    # real_data: [40, 32, 7680]
    # 数据分割
    data_divided = []
    for i in range(0, num * step, step):
        segment = real_data[:, :, i * fs:(i + window_size) * fs]  # [40, 32, 6s*fs]
        data_divided.append(segment)
    data_divided = np.vstack(data_divided)  # [40*num, 32, 6s*fs] [760, 32, 768]
    label_divided = np.vstack([label] * num)  # [40*num, 4]         [760, 4]
    return data_divided, label_divided


def data_process(data, labels):
    data, labels = data_divided(data[:, :32, :], labels)
    labels = label_process(labels)
    de_features = extract_feature(data)
    return de_features, labels


def coefficient(category_1, category_2, sample1_label, sample2_label):
    cls_bool1 = (sample1_label == category_1)
    cls_bool2 = (sample2_label == category_2)

    cls_bool1 = cls_bool1.view(-1, 1)
    cls_bool2 = cls_bool2.view(-1, 1)

    total_cls = torch.cat([cls_bool1, cls_bool2], dim=0).int()

    total_cls = total_cls.view(-1).int()

    total_coef = torch.ger(total_cls.cpu(), total_cls.cpu()).cuda()

    return total_coef


def lsd(source, target, source_label, target_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    intra_lsd_val = []
    inter_lsd_val = []
    num_class = 3

    for c1 in range(num_class):
        for c2 in range(num_class):
            coef_val = coefficient(c1, c2, source_label, target_label)
            e_ss = torch.div(coef_val[:n, :n] * XX, (coef_val[:n, :n]).sum() + 1e-5)
            e_st = torch.div(coef_val[:n, n:] * XY, (coef_val[:n, n:]).sum() + 1e-5)
            e_ts = torch.div(coef_val[n:, :n] * YX, (coef_val[n:, :n]).sum() + 1e-5)
            e_tt = torch.div(coef_val[n:, n:] * YY, (coef_val[n:, n:]).sum() + 1e-5)

            lsd_val = e_ss.sum() + e_tt.sum() - e_st.sum() - e_ts.sum()
            """if lsd_val.is_nan():
                    continue
            else:"""
            if c1 == c2:
                intra_lsd_val.append(lsd_val)
            elif c1 != c2:
                inter_lsd_val.append(lsd_val)

    loss = sum(intra_lsd_val) / len(intra_lsd_val) - sum(inter_lsd_val) / len(inter_lsd_val)
    return loss


def cosine_matrix(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    xty = torch.sum(x.unsqueeze(1) * y.unsqueeze(0), 2)
    return 1 - xty


def SM(Xs, Xt, Ys, Yt, Cs_memory, Ct_memory, Wt=None, decay=0.3):
    # Clone memory
    Cs = Cs_memory.clone()
    Ct = Ct_memory.clone()

    r = torch.norm(Xs, dim=1)[0]
    Ct = r * Ct / (torch.norm(Ct, dim=1, keepdim=True) + 1e-10)
    Cs = r * Cs / (torch.norm(Cs, dim=1, keepdim=True) + 1e-10)

    K = Cs.size(0)
    # for each class
    for k in range(K):
        Xs_k = Xs[Ys == k]
        Xt_k = Xt[Yt == k]

        if len(Xs_k) == 0:
            Cs_k = 0.0
        else:
            Cs_k = torch.mean(Xs_k, dim=0)

        if len(Xt_k) == 0:
            Ct_k = 0.0
        else:
            if Wt is None:
                Ct_k = torch.mean(Xt_k, dim=0)
            else:
                Wt_k = Wt[Yt == k]
                Ct_k = torch.sum(Wt_k.view(-1, 1) * Xt_k, dim=0) / (torch.sum(Wt_k) + 1e-5)

        Cs[k, :] = (1 - decay) * Cs_memory[k, :] + decay * Cs_k
        Ct[k, :] = (1 - decay) * Ct_memory[k, :] + decay * Ct_k

    Dist = cosine_matrix(Cs, Ct)

    return torch.sum(torch.diag(Dist)), Cs, Ct


def norminx(data):
    '''
    description: norm in x dimension
    param {type}:
        data: array
    return {type}
    '''
    for i in range(data.shape[0]):
        data[i] = normalization(data[i])
    return data


def norminy(data):
    dataT = data.T
    for i in range(dataT.shape[0]):
        dataT[i] = normalization(dataT[i])
    return dataT.T


def norminy_2d(data):
    data = data.reshape([-1, 5, 62])
    for j in range(5):
        for i in range(62):
            a = data[:, j, i]
            data[:, j, i] = normalization(data[:, j, i])
    return data.reshape([-1, 310])


def normalization(data):
    '''
    description:
    param {type}
    return {type}
    '''
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# package the data and label into one class


class CustomDataset(Dataset):
    # initialization: data and label
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    # get the size of data

    def __len__(self):
        return len(self.Data)

    # get the data and label

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        return data, label


# mmd loss and guassian kernel


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * 4)
    return loss


def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def get_number_of_label_n_trial(dataset_name):
    '''
    description: get the number of categories, trial number and the corresponding labels
    param {type}
    return {type}:
        trial: int
        label: int
        label_xxx: list 3*15
    '''
    # global variables
    label_seed4 = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                   [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2,
                    0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                   [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    label_seed3 = [[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]]

    if dataset_name == 'seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == 'seed4':
        label = 4
        trial = 24
        return trial, label, label_seed4
    elif dataset_name == 'deap':
        return
    else:
        print('Unexcepted dataset name')


def reshape_data(data, label):
    '''
    description: reshape data and initiate corresponding label vectors
    param {type}:
        data: list
        label: list
    return {type}:
        reshape_data: array, x*310
        reshape_label: array, x*1
    '''
    reshape_data = None
    reshape_label = None
    for i in range(len(data)):
        one_data = np.reshape(np.transpose(
            data[i], (1, 2, 0)), (-1, 310), order='F')
        one_label = np.full((one_data.shape[0], 1), label[i])
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    return reshape_data, reshape_label


def get_data_label_frommat(mat_path, dataset_name, session_id):
    '''
    description: load data from mat path and reshape to 851*310
    param {type}:
        mat_path: String
        session_id: int
    return {type}:
        one_sub_data, one_sub_label: array (851*310, 851*1)
    '''
    _, _, labels = get_number_of_label_n_trial(dataset_name)
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key: value for key,
    value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())
    one_sub_data, one_sub_label = reshape_data(mat_de_data, labels[session_id])
    return one_sub_data, one_sub_label


def sample_by_value(list, value, number):
    '''
    @Description: sample the given list randomly with given value
    @param {type}:
        list: list
        value: int {0,1,2,3}
        number: number of sampling
    @return:
        result_index: list
    '''
    result_index = []
    index_for_value = [i for (i, v) in enumerate(list) if v == value]
    result_index.extend(random.sample(index_for_value, number))
    return result_index


'''
For loading data
'''


def get_allmats_name(dataset_name):
    '''
    description: get the names of all the .mat files
    param {type}
    return {type}:
        allmats: list (3*15)
    '''
    path = dataset_path[dataset_name]
    sessions = os.listdir(path)
    sessions.sort()
    allmats = []
    for session in sessions:
        if session != '.DS_Store':
            mats = os.listdir(path + '/' + session)
            mats.sort()
            mats = mats[6:] + mats[:6]
            mats_list = []
            for mat in mats:
                mats_list.append(mat)
            allmats.append(mats_list)
    return path, allmats


def load_data(dataset_name):
    '''
    description: get all the data from one dataset
    param {type}
    return {type}:
        data: list 3(sessions) * 15(subjects), each data is x * 310
        label: list 3*15, x*1
    '''
    path, allmats = get_allmats_name(dataset_name)
    data = [([0] * 15) for i in range(3)]
    label = [([0] * 15) for i in range(3)]
    for i in range(len(allmats)):
        for j in range(len(allmats[0])):
            mat_path = path + '/' + str(i + 1) + '/' + allmats[i][j]
            one_data, one_label = get_data_label_frommat(
                mat_path, dataset_name, i)
            data[i][j] = one_data.copy()
            label[i][j] = one_label.copy()
    return np.array(data), np.array(label)


def get_one_data_and_label(data, label, session_id=1, subject_id=0):  # get subject0‘s data and label in session1
    one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    one_data, one_label = copy.deepcopy(one_session_data[subject_id]), copy.deepcopy(one_session_label[subject_id])
    # print("onedata:",one_data.shape,"onelable",one_label.shape)
    return one_data, one_label


def pick_one_data(dataset_name, session_id=1, cd_count=4, sub_id=0):
    '''
    @Description: pick one data from session 2 (or from other sessions),
    @param {type}:
        session_id: int
        cd_count: int (to indicate the number of calibration data)
    @return:
        832 for session 1, 851 for session 0
        cd_data: array (x*310, x is determined by cd_count)
        ud_data: array ((832-x)*310, the rest of that sub data)
        cd_label: array (x*1)
        ud_label: array ((832-x)*1)
    '''
    path, allmats = get_allmats_name(dataset_name)
    mat_path = path + "/" + str(session_id + 1) + \
               "/" + allmats[session_id][sub_id]
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key: value for key,
    value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())  # 24 * 62 * x * 5
    cd_list = []
    ud_list = []
    number_trial, number_label, labels = get_number_of_label_n_trial(
        dataset_name)
    session_label_one_data = labels[session_id]
    for i in range(number_label):
        # 根据给定的label值从label链表中拿到全部的index后根据数量随机采样
        cd_list.extend(sample_by_value(
            session_label_one_data, i, int(cd_count / number_label)))
    ud_list.extend([i for i in range(number_trial) if i not in cd_list])
    cd_label_list = copy.deepcopy(cd_list)
    ud_label_list = copy.deepcopy(ud_list)
    for i in range(len(cd_list)):
        cd_list[i] = mat_de_data[cd_list[i]]
        cd_label_list[i] = labels[session_id][cd_label_list[i]]
    for i in range(len(ud_list)):
        ud_list[i] = mat_de_data[ud_list[i]]
        ud_label_list[i] = labels[session_id][ud_label_list[i]]

    # reshape
    cd_data, cd_label = reshape_data(cd_list, cd_label_list)
    ud_data, ud_label = reshape_data(ud_list, ud_label_list)

    return cd_data, cd_label, ud_data, ud_label


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.01
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))

    return entropy


# calculate mutual information
def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy

    return 1 / Ixy


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return ((x1 - x2) ** 2).sum().sqrt()


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = (sx1 ** k).mean(0)
    ss2 = (sx2 ** k).mean(0)
    return l2diff(ss1, ss2)


class CMD(object):
    def __init__(self, n_moments=5):
        self.n_moments = n_moments

    def forward(self, x1, x2):
        mx1 = x1.mean(dim=0)
        mx2 = x2.mean(dim=0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        scms = l2diff(mx1.detach(), mx2.detach())  # detach for avoid inplace gradient operation

        for i in range(self.n_moments - 1):
            # moment diff of centralized samples
            scms += moment_diff(sx1, sx2, i + 2)
        return scms


def load_deap(data_name):
    deap_path = dataset_path[data_name]
    dat_data = [0] * 32
    dat_label = [0] * 32
    for i, index in enumerate(os.listdir(deap_path)):
        with open(os.path.join(deap_path, index), 'rb') as f:
            x = cPickle.load(f, encoding='iso-8859-1')
            eeg_data = x['data'][:, :32, :]
            label_all = x['labels']
            data, label = data_divided(eeg_data[:, :32, :], label_all)
            one_labels = label_process(label)
            one_labels = np.expand_dims(one_labels, axis=-1)
            one_de_features = extract_feature(data)
        dat_data[i] = one_de_features.reshape(-1, 160).copy()
        dat_label[i] = np.array(one_labels).copy()
        # dat_data[i] = one_de_features.copy()
        # dat_label[i] = np.array(one_labels).copy()
    np.array(dat_data), np.array(dat_label)
    return np.array(dat_data), np.array(dat_label)


def get_deap():
    with open('deap_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('deap_label.pkl', 'rb') as f:
        label = pickle.load(f)
    return data, label


if __name__ == '__main__':
    data, label = load_deap('deap')
    print(data.shape, label.shape)
    # 保存数据
    with open('deap_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('deap_label.pkl', 'wb') as f:
        pickle.dump(label, f)
