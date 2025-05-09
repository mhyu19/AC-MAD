import torch.nn.functional as F
import torch
import utils as utils
from network import WassersteinDiscriminatorSN
import torch.nn as nn


class MultiheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        # K: [64,5,62], batch_size 为 64，有 5 个频带，每个向量是 62 维
        # V: [64,5,62]
        # Q: [64,5,62]
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        # 第 1 步：Q 乘以 K的转置，除以scale
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        attention = self.do(torch.softmax(attention, dim=-1))
        # 第三步，attention结果与V相乘，得到多头注意力的结果
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x, attention


class CFE(nn.Module):
    def __init__(self, in_planes=None):
        super(CFE, self).__init__()
        if in_planes is None:
            in_planes = [5, 62]
        self.band_attention = MultiheadAttention(hid_dim=in_planes[1], n_heads=in_planes[1], dropout=0.1)
        self.channel_attention = MultiheadAttention(hid_dim=in_planes[0], n_heads=in_planes[0], dropout=0.1)
        self.module = nn.Sequential(
            nn.Linear(310, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), 5, 62)
        residual = x
        out, att_weight1 = self.band_attention(x, x, x)  # out.shape=(B, 5, 62)
        out = out.transpose(1, 2)
        out, att_weight2 = self.channel_attention(out, out, out)  # out.shape=(B, 52, 5)
        out = out.transpose(1, 2)
        out = out.reshape(out.size(0), -1)
        out = out + residual.view(residual.size(0), -1)
        out = self.module(out)
        # out = self.module(x)
        return out, [att_weight1, att_weight2]


def pretrained_CFE(pretrained=False):
    model = CFE()
    if pretrained:
        pass
    return model


class DSFE(nn.Module):
    def __init__(self):
        super(DSFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


class ADS_MSDANet(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=4, domain_weight=None):
        super(ADS_MSDANet_TSNE, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        if domain_weight is None:
            domain_weight = []
        self.domain_weight = torch.Tensor(domain_weight)
        self.adv_net = WassersteinDiscriminatorSN(32, 64).cuda()  # domain_discriminator
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')
        self.number_of_source = number_of_source
        self.id = id
        self.weight_d = torch.Tensor(domain_weight).cuda()
        # self.weight_d += 1

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        """
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
            之所以target data每一条线都要过一遍是因为要计算discrepency loss, mmd和cls都只要mark-th那条线就行
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type}
        """
        mmd_loss = 0
        wd_loss = 0
        data_src_DSFE = []
        data_tgt_DSFE = []
        lsd_loss = 0
        cls_loss = 0
        tcls_loss = 0
        loss_mmd = []
        if self.training == True:
            # common feature extractor
            data_src_CFE, attention_weights_src = self.sharedNet(data_src)
            # data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE, attention_weights_tgt = self.sharedNet(data_tgt)
            # data_tgt_CFE = self.sharedNet(data_tgt)

            data_src_CFE = torch.chunk(data_src_CFE, number_of_source, 0)
            label_src = torch.chunk(label_src, number_of_source, 0)
            # att_src_CFE_channels = torch.chunk(attention_weights_src[-1], number_of_source, 0)
            # att_src_CFE_bands = torch.chunk(attention_weights_src[-2], number_of_source, 0)
            pred_tgt = []
            with torch.no_grad():
                for i in range(number_of_source):
                    DSFE_name = 'self.DSFE' + str(i)
                    data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                    DSC_name = 'self.cls_fc_DSC' + str(i)
                    pred_tgt_i = eval(DSC_name)(data_tgt_DSFE_i)
                    pred_tgt_i = F.softmax(pred_tgt_i, dim=1)
                    pred_tgt.append(pred_tgt_i.unsqueeze(1))
                pred_tgt = torch.cat(pred_tgt, dim=1)
                pred_tgt_w = pred_tgt.mean(1)
                max_prob, label_tgt = pred_tgt_w.max(1)  # (B)
                label_tgt_mask = (max_prob >= 0.95).float()

            for i in range(number_of_source):
                # Each domian specific feature extractor
                # to extract the domain specific feature of target data
                DSFE_name = 'self.DSFE' + str(i)
                data_src_DSFE_i = eval(DSFE_name)(data_src_CFE[i])
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                out_src = self.adv_net(data_src_DSFE_i)
                out_tgt = self.adv_net(data_tgt_DSFE_i)
                wdist = torch.abs((out_src.mean() - out_tgt.mean()))
                # wd_loss += (self.domain_weight[i] * wdist)
                wd_loss += wdist
                data_tgt_DSFE.append(data_src_DSFE_i)
                data_src_DSFE.append(data_tgt_DSFE_i)

                # mmd_loss += (self.domain_weight[i] * (utils.mmd_linear(data_src_DSFE_i, data_tgt_DSFE_i)))
                mmd_loss = (utils.mmd_linear(data_src_DSFE_i, data_tgt_DSFE_i))
                loss_mmd.append(mmd_loss)

                # Each domian specific classifier
                DSC_name = 'self.cls_fc_DSC' + str(i)
                pred_src_i = eval(DSC_name)(data_src_DSFE_i)
                cls_loss += F.nll_loss(F.log_softmax(
                    pred_src_i, dim=1), label_src[i].squeeze())

                pred_tgt_i = eval(DSC_name)(data_tgt_DSFE_i)
                tcls_loss_i = F.nll_loss(F.log_softmax(
                    pred_tgt_i, dim=1), label_tgt, reduction='none')
                tcls_loss += (tcls_loss_i * label_tgt_mask).mean()

                max_prob, pseudo_label = torch.max(F.softmax(pred_src_i, dim=1), dim=1)
                confident_bool = max_prob >= 0.60
                confident_example = data_tgt_DSFE[mark][confident_bool]
                confident_label = pseudo_label[confident_bool]
                lsd_loss += utils.lsd(data_src_DSFE_i, confident_example, label_src[i], confident_label)
            alpha = 0.9
            weights_dom = torch.stack(loss_mmd)
            weights_dom = weights_dom.detach()
            weights_dom = 1 / weights_dom
            weights_dom = F.softmax(weights_dom, dim=0)
            # import pdb;pdb.set_trace()
            self.weight_d = (1 - alpha) * self.weight_d + alpha * weights_dom
            align_loss = 0
            for domain_idx in range(number_of_source):
                align_loss += self.weight_d[domain_idx] * loss_mmd[domain_idx]

            return cls_loss + 0.2 * tcls_loss, align_loss, wd_loss, lsd_loss, attention_weights_src, attention_weights_tgt

        else:
            data_CFE, att_weights = self.sharedNet(data_src)
            pred = []
            feature_DSFE = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                feature_DSFE.append(feature_DSFE_i)
                pred.append(eval(DSC_name)(feature_DSFE_i))

            return pred, data_CFE, feature_DSFE


class ADS_MSDANet_TSNE(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=4, domain_weight=None):
        super(ADS_MSDANet_TSNE, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        if domain_weight is None:
            domain_weight = []
        self.domain_weight = torch.Tensor(domain_weight)
        self.adv_net = WassersteinDiscriminatorSN(32, 64).cuda()  # domain_discriminator
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')
        self.number_of_source = number_of_source
        self.id = id
        self.weight_d = torch.Tensor(domain_weight).cuda()
        # self.weight_d += 1

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        """
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
            之所以target data每一条线都要过一遍是因为要计算discrepency loss, mmd和cls都只要mark-th那条线就行
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type}
        """
        mmd_loss = 0
        wd_loss = 0
        data_src_DSFE = []
        data_tgt_DSFE = []
        lsd_loss = 0
        cls_loss = 0
        tcls_loss = 0
        loss_mmd = []
        if self.training == True:
            # common feature extractor
            data_src_CFE, attention_weights_src = self.sharedNet(data_src)
            # data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE, attention_weights_tgt = self.sharedNet(data_tgt)
            # data_tgt_CFE = self.sharedNet(data_tgt)

            data_src_CFE = torch.chunk(data_src_CFE, number_of_source, 0)
            label_src = torch.chunk(label_src, number_of_source, 0)
            # att_src_CFE_channels = torch.chunk(attention_weights_src[-1], number_of_source, 0)
            # att_src_CFE_bands = torch.chunk(attention_weights_src[-2], number_of_source, 0)
            pred_tgt = []
            with torch.no_grad():
                for i in range(number_of_source):
                    DSFE_name = 'self.DSFE' + str(i)
                    data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                    DSC_name = 'self.cls_fc_DSC' + str(i)
                    pred_tgt_i = eval(DSC_name)(data_tgt_DSFE_i)
                    pred_tgt_i = F.softmax(pred_tgt_i, dim=1)
                    pred_tgt.append(pred_tgt_i.unsqueeze(1))
                pred_tgt = torch.cat(pred_tgt, dim=1)
                pred_tgt_w = pred_tgt.mean(1)
                max_prob, label_tgt = pred_tgt_w.max(1)  # (B)
                label_tgt_mask = (max_prob >= 0.95).float()

            for i in range(number_of_source):
                # Each domian specific feature extractor
                # to extract the domain specific feature of target data
                DSFE_name = 'self.DSFE' + str(i)
                data_src_DSFE_i = eval(DSFE_name)(data_src_CFE[i])
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                out_src = self.adv_net(data_src_DSFE_i)
                out_tgt = self.adv_net(data_tgt_DSFE_i)
                wdist = torch.abs((out_src.mean() - out_tgt.mean()))
                # wd_loss += (self.domain_weight[i] * wdist)
                wd_loss += wdist
                data_tgt_DSFE.append(data_src_DSFE_i)
                data_src_DSFE.append(data_tgt_DSFE_i)

                # mmd_loss += (self.domain_weight[i] * (utils.mmd_linear(data_src_DSFE_i, data_tgt_DSFE_i)))
                mmd_loss = (utils.mmd_linear(data_src_DSFE_i, data_tgt_DSFE_i))
                loss_mmd.append(mmd_loss)

                # Each domian specific classifier
                DSC_name = 'self.cls_fc_DSC' + str(i)
                pred_src_i = eval(DSC_name)(data_src_DSFE_i)
                cls_loss += F.nll_loss(F.log_softmax(
                    pred_src_i, dim=1), label_src[i].squeeze())

                pred_tgt_i = eval(DSC_name)(data_tgt_DSFE_i)
                tcls_loss_i = F.nll_loss(F.log_softmax(
                    pred_tgt_i, dim=1), label_tgt, reduction='none')
                tcls_loss += (tcls_loss_i * label_tgt_mask).mean()

                max_prob, pseudo_label = torch.max(F.softmax(pred_src_i, dim=1), dim=1)
                confident_bool = max_prob >= 0.60
                confident_example = data_tgt_DSFE[mark][confident_bool]
                confident_label = pseudo_label[confident_bool]
                lsd_loss += utils.lsd(data_src_DSFE_i, confident_example, label_src[i], confident_label)
            alpha = 0.9
            weights_dom = torch.stack(loss_mmd)
            weights_dom = weights_dom.detach()
            weights_dom = 1 / weights_dom
            weights_dom = F.softmax(weights_dom, dim=0)
            # import pdb;pdb.set_trace()
            self.weight_d = (1 - alpha) * self.weight_d + alpha * weights_dom
            align_loss = 0
            for domain_idx in range(number_of_source):
                align_loss += self.weight_d[domain_idx] * loss_mmd[domain_idx]

            return cls_loss + 0.2 * tcls_loss, align_loss, wd_loss, lsd_loss, attention_weights_src, attention_weights_tgt

        else:
            data_CFE, att_weights = self.sharedNet(data_src)
            pred = []
            feature_DSFE = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                feature_DSFE.append(feature_DSFE_i)
                pred.append(eval(DSC_name)(feature_DSFE_i))

            return pred, data_CFE, feature_DSFE
