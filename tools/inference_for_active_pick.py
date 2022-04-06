import json
import numpy as np
import torch
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data import detection_utils as utils

from activeteacher.config import add_activeteacher_config
from activeteacher.engine.trainer import ActiveTeacherTrainer
from activeteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel



@torch.no_grad()
def uncertainty_entropy(p):
    # p.size() = num_instances of a image, num_classes
    p = F.softmax(p, dim=1)
    p = - torch.log2(p) * p
    entropy_instances = torch.sum(p, dim=1)
    # set uncertainty of image eqs the mean uncertainty of instances
    entropy_image = torch.mean(entropy_instances)
    return entropy_image


data_hook = {}
def box_predictor_hooker(m, i, o):
    data_hook['scores_hooked'] = o[0].clone().detach()
    data_hook['boxes_hooked'] = o[1].clone().detach()


def setup(args):
    """
        Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_activeteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    Trainer = ActiveTeacherTrainer
    assert args.eval_only is True, "Inference should be eval only."
    inference(Trainer, cfg)


@torch.no_grad()
def inference(Trainer, cfg):
    print('Loading Model named: ', cfg.MODEL.WEIGHTS)
    model = Trainer.build_model(cfg)
    model_teacher = Trainer.build_model(cfg)
    ensem_ts_model = EnsembleTSModel(model_teacher, model)

    DetectionCheckpointer(
        ensem_ts_model, save_dir=cfg.OUTPUT_DIR
    ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)


    ensem_ts_model.modelTeacher.roi_heads.box_predictor.register_forward_hook(box_predictor_hooker)
    ensem_ts_model.modelTeacher.eval()
    ensem_ts_model.modelTeacher.training = False
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dic={}
    from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
    for j,item in enumerate(dataset_dicts):
        file_name = item['file_name']
        print(j,file_name)
        image = utils.read_image(file_name, format='BGR')
        image = torch.from_numpy(image.copy()).permute(2,0,1)
        res = ensem_ts_model.modelTeacher.inference([{'image':image}])

        # score
        scores = data_hook['scores_hooked'].to(torch.device("cpu"))
        entropy = uncertainty_entropy(scores)

        dic[file_name]=[]
        for i in range(len(res[0]['instances'])):
            box_info = {'confidence score':np.float(res[0]['instances'].scores.cpu().detach().numpy()[i]),
                        'pred class':np.int(res[0]['instances'].pred_classes.cpu().detach().numpy()[i]),
                        'pred box':res[0]['instances'].pred_boxes.tensor[i].cpu().detach().numpy().tolist(),
                        'entropy': entropy.cpu().detach().clone().item()
                        }
            dic[file_name].append(box_info)

        del res
        del image
        del data_hook['scores_hooked']
        del data_hook['boxes_hooked']
        torch.cuda.empty_cache()

    with open(FILE_PATH, 'w') as f:
        f.write(json.dumps(dic))


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument("--static-file",type=str,default='temp/coco/static_by_random10.json') #Json file of the intermediate process
    parser.add_argument("--model-weights",type=str,default='output/model_best.pth')
    args = parser.parse_args()
    args.eval_only = True
    args.resume = True
    args.num_gpus = 1
    FILE_PATH = args.static_file 
    #args.config_file = 'configs/coco/faster_rcnn_R_50_FPN_sup10_run1.yaml' #the config file you used to train this inference model

    # you should config MODEL.WEIGHTS and keep other hyperparameters default(Odd-numbered items are keys, even-numbered items are values)
    args.opts = ['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5,'MODEL.ROI_HEADS.NMS_THRESH_TEST', 0.5,
     'TEST.DETECTIONS_PER_IMAGE', 20, 'INPUT.FORMAT', 'RGB','MODEL.WEIGHTS',args.model_weights]
    print("Command Line Args:", args)
    main(args)