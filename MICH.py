from setting import *
from tnet import *

from ops import *
from utils.calc_hammingranking import calc_map
import os
import time
import scipy.io as sio
from tqdm import tqdm
import torch


class MICH(object):
    def __init__(self):
        # include hyper parameters
        self.hyper_mi = 50
        self.hyper_sigma = 0.01
        self.alpha_v = 0.01
        self.alpha_t = 1 - self.alpha_v
        self.eta = 100
        self.gamma = 0.01

        self.train_L = train_L
        self.train_X = train_x
        self.train_Y = train_y

        self.query_L = query_L
        self.query_X = query_x
        self.query_Y = query_y

        self.retrieval_L = retrieval_L
        self.retrieval_X = retrieval_x
        self.retrieval_Y = retrieval_y

        self.lr_lab = lr_lab
        self.lr_img = lr_img
        self.lr_txt = lr_txt
        self.Sim = Sim

        self.lnet = LabelNet().cuda()
        self.inet = ImageNet().cuda()
        self.tnet = TextNet().cuda()

        self.checkpoint_dir = checkpoint_path
        self.bit = bit
        self.num_train = num_train
        self.SEMANTIC_EMBED = SEMANTIC_EMBED

    def train(self):
        self.lnet_opt = torch.optim.Adam(self.lnet.parameters(), lr=self.lr_lab[0])
        self.inet_opt = torch.optim.Adam(self.inet.parameters(), lr=self.lr_img[0])
        self.tnet_opt = torch.optim.Adam(self.tnet.parameters(), lr=self.lr_txt[0])

        var = {}
        var['lr_lab'] = self.lr_lab
        var['lr_img'] = self.lr_img
        var['lr_txt'] = self.lr_txt

        var['batch_size'] = batch_size
        var['F'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['G'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['H'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['FG'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED).astype(np.float32)
        var['B'] = np.sign(self.alpha_v * var['F'] + self.alpha_t * var['G'] + self.eta * var['H'])

        # Iterations
        for epoch in range(Epoch):
            results = {}
            results['loss_labNet'] = []
            results['loss_imgNet'] = []
            results['loss_txtNet'] = []
            results['Loss_D'] = []
            results['mapl2l'] = []
            results['mapi2i'] = []
            results['mapt2t'] = []

            print('++++++++Start train lab_net++++++++')
            for idx in range(2):
                lr_lab_Up = var['lr_lab'][epoch:]
                lr_lab = lr_lab_Up[idx]
                for train_labNet_k in range(k_lab_net // (idx + 1)):
                    adjust_learning_rate(self.lnet_opt, lr_lab)
                    train_labNet_loss = self.train_lab_net(var)
                    var['B'] = np.sign(self.alpha_v * var['F'] + self.alpha_t * var['G'] + self.eta * var['H'])
                    # train_labNet_loss = self.calc_labnet_loss(var['H'], var['F'], var['G'], var['B'], Sim)
                    results['loss_labNet'].append(train_labNet_loss)
                    print('---------------------------------------------------------------')
                    print('...epoch: %3d, loss_labNet: %3.3f' % (epoch, train_labNet_loss))
                    print('---------------------------------------------------------------')
                    if train_labNet_k > 1 and (results['loss_labNet'][-1] - results['loss_labNet'][-2]) >= 0:
                        break

            print('++++++++Starting Train txt_net++++++++')
            for idx in range(3):
                lr_txt_Up = var['lr_txt'][epoch:]
                lr_txt = lr_txt_Up[idx]
                for train_txtNet_k in range(k_txt_net // (idx + 1)):
                    adjust_learning_rate(self.tnet_opt, lr_txt)
                    train_txtNet_loss = self.train_txt_net(var)
                    var['B'] = np.sign(self.alpha_v * var['F'] + self.alpha_t * var['G'] + self.eta * var['H'])
                    if train_txtNet_k % 2 == 0:
                        # train_txtNet_loss = self.calc_loss(self.alpha_t, var['G'], var['H'], var['B'], Sim)
                        results['loss_txtNet'].append(train_txtNet_loss)
                        print('---------------------------------------------------------------')
                        print('...epoch: %3d, Loss_txtNet: %s' % (epoch, train_txtNet_loss))
                        print('---------------------------------------------------------------')
                    if train_txtNet_k > 2 and (results['loss_txtNet'][-1] - results['loss_txtNet'][-2]) >= 0:
                        break

            print('++++++++Starting Train img_net++++++++')
            for idx in range(3):
                lr_img_Up = var['lr_img'][epoch:]
                lr_img = lr_img_Up[idx]
                for train_imgNet_k in range(k_img_net // (idx + 1)):
                    adjust_learning_rate(self.inet_opt, lr_img)
                    train_imgNet_loss = self.train_img_net(var)
                    var['B'] = np.sign(self.alpha_v * var['F'] + self.alpha_t * var['G'] + self.eta * var['H'])
                    if train_imgNet_k % 2 == 0:
                        # train_imgNet_loss = self.calc_loss(self.alpha_v, var['F'], var['H'], var['B'], Sim)
                        results['loss_imgNet'].append(train_imgNet_loss)
                        print('---------------------------------------------------------------')
                        print('...epoch: %3d, loss_imgNet: %3.3f' % (epoch, train_imgNet_loss))
                        print('---------------------------------------------------------------')
                    if train_imgNet_k > 2 and (results['loss_imgNet'][-1] - results['loss_imgNet'][-2]) >= 0:
                        break

            '''
            evaluation after each epoch
            '''
            with torch.no_grad():
                qBY = self.generate_code(self.query_Y, "text")
                rBY = self.generate_code(self.retrieval_Y, "text")
                qBX = self.generate_code(self.query_X, "image")
                rBX = self.generate_code(self.retrieval_X, "image")

                mapi2t = calc_map(qBX, rBY, self.query_L, self.retrieval_L)
                mapt2i = calc_map(qBY, rBX, self.query_L, self.retrieval_L)
                mapi2i = calc_map(qBX, rBX, self.query_L, self.retrieval_L)
                mapt2t = calc_map(qBY, rBY, self.query_L, self.retrieval_L)

                condition_dir = './result-mi-%f-sigma-%f' % (self.hyper_mi, self.hyper_sigma)
                if not os.path.exists(condition_dir):
                    os.mkdir(condition_dir)

                save_dir_name = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
                cur_dir_path = os.path.join(condition_dir, save_dir_name)
                os.mkdir(cur_dir_path)

                scipy.io.savemat(os.path.join(cur_dir_path, 'B_all.mat'), {
                    'BxTest': qBX,
                    'BxTrain': rBX,
                    'ByTest': qBY,
                    'ByTrain': rBY,
                    'LTest': self.query_L,
                    'LTrain': self.retrieval_L
                })

                with open(os.path.join(cur_dir_path, 'map.txt'), 'a') as f:
                    f.write('==================================================\n')
                    f.write('...test map: map(i->t): %3.3f, map(t->i): %3.3f\n' % (mapi2t, mapt2i))
                    f.write('...test map: map(t->t): %3.3f, map(i->i): %3.3f\n' % (mapt2t, mapi2i))
                    f.write('==================================================\n')

                '''
                save checkpoint
                '''
                state = {
                    'lnet': self.lnet.state_dict(),
                    'inet': self.inet.state_dict(),
                    'tnet': self.tnet.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, os.path.join(cur_dir_path, self.checkpoint_path))

    def train_lab_net(self, var):
        print('update label_net')
        F = var['F']
        G = var['G']
        H = var['H']
        B = var['B']
        loss_total = 0.0
        num_train = self.train_L.shape[0]
        for iter in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = self.train_L[ind, :]
            label = self.train_L[ind, :].astype(np.float32)
            label = label.reshape([label.shape[0], 1, 1, label.shape[1]])
            S = calc_neighbor(self.train_L, sample_L)
            hsh_L = self.lnet(torch.from_numpy(label).cuda())

            H[ind, :] = hsh_L.detach().cpu().numpy()
            S_cuda = torch.from_numpy(S).cuda()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            theta_FL = 1.0 / 2 * torch.from_numpy(F).cuda().mm(hsh_L.transpose(1, 0))
            Loss_pair_Hsh_FL = nn.functional.mse_loss(S_cuda.mul(theta_FL), nn.functional.softplus(theta_FL),
                                                      reduction='sum')
            theta_GL = 1.0 / 2 * torch.from_numpy(G).cuda().mm(hsh_L.transpose(1, 0))
            Loss_pair_Hsh_GL = nn.functional.mse_loss(S_cuda.mul(theta_GL), nn.functional.softplus(theta_GL),
                                                      reduction='sum')
            Loss_quant_L = nn.functional.mse_loss(B_cuda, hsh_L, reduction='sum')
            loss_l = (Loss_pair_Hsh_FL + Loss_pair_Hsh_GL) + self.eta * Loss_quant_L
            loss_total += float(loss_l.detach().cpu().numpy())

            self.lnet_opt.zero_grad()
            loss_l.backward()
            self.lnet_opt.step()
        return loss_total

    def train_img_net(self, var):
        print('update image_net')
        F = var['F']
        H = var['H']
        FG = var['FG']
        B = var['B']
        loss_total = 0.0
        num_train = self.train_X.shape[0]
        for iter in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = train_L[ind, :]
            image = self.train_X[ind, :, :, :].astype(np.float32)
            S = calc_neighbor(train_L, sample_L)
            fea_I, hsh_I, lab_I, fea_T_pred, mu_I, log_sigma_I = self.inet(torch.from_numpy(image).cuda())
            F[ind, :] = hsh_I.detach().cpu().numpy()
            fea_T_real = torch.from_numpy(FG[ind, :]).cuda()
            S_cuda = torch.from_numpy(S).cuda()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            theta_MH = 1.0 / 2 * torch.from_numpy(H).cuda().mm(hsh_I.transpose(1, 0))
            Loss_pair_Hsh_MH = nn.functional.mse_loss(S_cuda.mul(theta_MH), nn.functional.softplus(theta_MH),
                                                      reduction='sum')
            theta_MM = 1.0 / 2 * torch.from_numpy(F).cuda().mm(hsh_I.transpose(1, 0))
            Loss_pair_Hsh_MM = nn.functional.mse_loss(S_cuda.mul(theta_MM), nn.functional.softplus(theta_MM),
                                                      reduction='sum')
            Loss_quant_I = nn.functional.mse_loss(B_cuda, hsh_I, reduction='sum')
            Loss_label_I = nn.functional.mse_loss(torch.from_numpy(self.train_L[ind, :]).cuda(), lab_I, reduction='sum')
            Loss_prior_kl = torch.sum(mu_I.pow(2).add_(log_sigma_I.exp()).mul_(-1).add_(1).add_(log_sigma_I)).mul_(-0.5)
            Loss_cross_hash_MI = nn.functional.binary_cross_entropy_with_logits(fea_T_pred, torch.sigmoid(fea_T_real), reduction='sum') \
                                 + self.hyper_sigma * Loss_prior_kl

            loss_i = (Loss_pair_Hsh_MH + Loss_pair_Hsh_MM) \
                + self.alpha_v * Loss_quant_I \
                + self.gamma * Loss_label_I \
                + self.hyper_mi * Loss_cross_hash_MI
            loss_total += float(loss_i.detach().cpu().numpy())

            self.inet_opt.zero_grad()
            loss_i.backward()
            self.inet_opt.step()
        return loss_total

    def train_txt_net(self, var):
        print('update text_net')
        G = var['G']
        H = var['H']
        FG = var['FG']
        B = var['B']
        loss_total = 0.0
        num_train = self.train_Y.shape[0]
        for iter in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = train_L[ind, :]
            text = self.train_Y[ind, :].astype(np.float32)
            text = text.reshape([text.shape[0], 1, text.shape[1], 1])
            S = calc_neighbor(train_L, sample_L)
            fea_T, hsh_T, lab_T = self.tnet(torch.from_numpy(text).cuda())
            G[ind, :] = hsh_T.detach().cpu().numpy()
            FG[ind, :] = fea_T.detach().cpu().numpy()
            S_cuda = torch.from_numpy(S).cuda()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            theta_MH = 1.0 / 2 * torch.from_numpy(H).cuda().mm(hsh_T.transpose(1, 0))
            Loss_pair_Hsh_MH = nn.functional.mse_loss(S_cuda.mul(theta_MH), nn.functional.softplus(theta_MH),
                                                      reduction='sum')
            theta_MM = 1.0 / 2 * torch.from_numpy(G).cuda().mm(hsh_T.transpose(1, 0))
            Loss_pair_Hsh_MM = nn.functional.mse_loss(S_cuda.mul(theta_MM), nn.functional.softplus(theta_MM),
                                                      reduction='sum')
            Loss_quant_T = nn.functional.mse_loss(B_cuda, hsh_T, reduction='sum')
            Loss_label_T = nn.functional.mse_loss(torch.from_numpy(self.train_L[ind, :]).cuda(), lab_T, reduction='sum')
            loss_t = (Loss_pair_Hsh_MH + Loss_pair_Hsh_MM) \
                + self.alpha_t * Loss_quant_T \
                + self.gamma * Loss_label_T
            loss_total += float(loss_t.detach().cpu().numpy())
            self.tnet_opt.zero_grad()
            loss_t.backward()
            self.tnet_opt.step()
        return loss_total

    def calc_isfrom_acc(self, train_isfrom_, Train_ISFROM):
        erro = Train_ISFROM.shape[0] - np.sum(
            np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype(int))
        acc = np.divide(np.sum(np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype('float32')),
                        Train_ISFROM.shape[0])
        return erro, acc

    def generate_code(self, modal, generate):
        batch_size = 128
        if generate == "label":
            num_data = modal.shape[0]
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(range(num_data // batch_size + 1)):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                label = modal[ind, :].astype(np.float32)
                Fea_L, Hsh_L, Lab_L = self.lnet(torch.from_numpy(label).cuda())
                B[ind, :] = Hsh_L.detach().cpu().numpy()
        elif generate == "image":
            num_data = len(modal)
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(range(num_data // batch_size + 1)):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                image = modal[ind, :, :, :].astype(np.float32)
                # image = image - meanpix.astype(np.float32)
                Fea_I, Hsh_I, Lab_I, fea_T_pred, mu_I, log_sigma_I = self.inet(torch.from_numpy(image).cuda())
                B[ind, :] = Hsh_I.detach().cpu().numpy()
        else:
            num_data = modal.shape[0]
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(range(num_data // batch_size + 1)):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                text = modal[ind, :].astype(np.float32)
                Fea_T, Hsh_T, Lab_T = self.tnet(torch.from_numpy(text).cuda())
                B[ind, :] = Hsh_T.detach().cpu().numpy()
        B = np.sign(B)
        return B

    def calc_labnet_loss(self, H, F, G, B, Sim):
        theta_fh = np.matmul(F, np.transpose(H)) / 2
        term_fh = np.sum(nn.functional.softplus(torch.from_numpy(theta_fh)).numpy() - Sim * theta_fh)
        theta_gh = np.matmul(G, np.transpose(H)) / 2
        term_gh = np.sum(nn.functional.softplus(torch.from_numpy(theta_gh)).numpy() - Sim * theta_gh)
        term_quant = np.sum(np.power(B - H, 2))
        loss = (term_fh + term_gh) + self.eta * term_quant
        print('pairwise_hash_FH:', term_fh)
        print('pairwise_hash_GH:', term_gh)
        print('quant loss:', term_quant)
        return loss

    def calc_loss(self, alpha, M, H, B, Sim):
        theta_mh = np.matmul(M, np.transpose(H)) / 2
        term_mh = np.sum(nn.functional.softplus(torch.from_numpy(theta_mh)).numpy() - Sim * theta_mh)
        theta_mm = np.matmul(M, np.transpose(M)) / 2
        term_mm = np.sum(nn.functional.softplus(torch.from_numpy(theta_mm)).numpy() - Sim * theta_mm)
        term_quant = np.sum(np.power(B - M, 2))
        loss = (term_mh + term_mm) + alpha * term_quant
        print('pairwise_hash_MH:', term_mh)
        print('pairwise_hash_MM:', term_mm)
        print('quant loss:', term_quant)
        return loss
