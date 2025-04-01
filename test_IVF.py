from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import numpy as np

from newmodel import freFuse
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/CDDFuse_fre.pth"
# for dataset_name in ["TNO","RoadScene","MSRS"]:# 加载模型吗？
for dataset_name in ["MSRS"]:
    print("\n"*2+"="*80)
    model_name="CDDFuse    "
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name) 
    test_out_folder=os.path.join('test_result',dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    fus = nn.DataParallel(freFuse()).to(device)
    # Encoder = Restormer_Encoder().to(device)
    # Decoder = Restormer_Decoder().to(device)

    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)


    # BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    # DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])#加载模型参数
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])

    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])#加载模型参数
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    fus.load_state_dict(torch.load(ckpt_path)['fre'])# 这样行吗？

    Encoder.eval()
    Decoder.eval()

    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):

            # data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            # data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            #以灰度读取图像，将每个图片进行拓展维度，并归一化
            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[np.newaxis, np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='GRAY')[np.newaxis, np.newaxis, ...]/255.0

            # img = img.astype(np.uint8)

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)

            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)

            frefus, amp, pha = fus(data_VIS, data_IR)
            # 损失函数也加一下吧
            # （1，64，128，128）
            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D, frefus) # 这里加上频率分支
            #对输出进行归一化
            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = fi.astype(np.uint8)#转换为uint8,后来新加的
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)


    eval_folder=test_out_folder  
    ori_img_folder=test_folder
#将测试结果进行评价
    metric_result = np.zeros((8))
    for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))
            )
    print("="*80)