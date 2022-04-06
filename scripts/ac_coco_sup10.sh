mkdir temp
mkdir temp/coco
mkdir results
mkdir results/coco
mkdir dataseed/coco_pick
#1、Train a pick model on 5% random data
python tools/train_net.py \
      --num-gpus 8 \
      --config configs/coco/faster_rcnn_R_50_FPN_sup5_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 OUTPUT_DIR output/coco/faster_rcnn_R_50_FPN_sup5_run1_16bs

#2、Use the trained model from step 1 to get the indicator file of the dataset
python tools/inference_for_active_pick.py \
    --static-file temp/coco/static_by_random5.json \
    --model-weights output/coco/faster_rcnn_R_50_FPN_sup5_run1_16bs/model_best.pth \
    --config configs/coco/faster_rcnn_R_50_FPN_sup5_run1.yaml \

python tools/active_pick_evaluation.py \
    --static-file temp/coco/static_by_random5.json \
    --indicator-file results/coco/5random_maxnorm

#3、Use the indictor file from step 2 to generate pick data and merge random data
python tools/generate_pick_merge_random_data_partition.py \
    --random-file dataseed/COCO_supervision.txt \
    --random-percent 5.0 \
    --indicator-file results/coco/5random_maxnorm.txt \
    --pick-percent 5.0 \
    --reverse True \
    --save-file dataseed/coco_pick/pick_maxnorm5+random5.txt

#4、Train a model from scratch using the 10% data partition from step 3
python tools/train_net.py \
      --num-gpus 8 \
      --config configs/coco/faster_rcnn_R_50_FPN_sup10_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 OUTPUT_DIR output/coco/faster_rcnn_R_50_FPN_sup10_run1_16bs DATALOADER.RANDOM_DATA_SEED_PATH dataseed/coco_pick/pick_maxnorm5+random5.txt
