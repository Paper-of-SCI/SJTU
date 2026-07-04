```bash
cd /home/leo/Projects/SJTU/methods/external/gsplat/examples
python leo_3dgs_trainer.py default \
  --data_dir /home/leo/Projects/SJTU/src/datasets/SeathruNeRF_dataset/JapaneseGradens-RedSea/undistorted_pinhole \
  --data_factor 2 \
  --result_dir ./results/JapaneseGradens-RedSea-underwaterRasterizeFormula \
  --disable_viewer \
  --disable_video \
  --batch_size 1 \
  --eval_steps 1000 22000 29000 30000 \
  --save_steps 15000 30000 \
  --use_gae_encoder
```

# 单用物理渲染公式

```Shell
--use_underwater_rasterize_formula \
--depth-loss
```

# 单用depthLoss

```Shell
--depth-loss \
--use_depth_loss
```

# 物理渲染 AND depthLoss

```Shell
--use_underwater_rasterize_formula \
--depth-loss \
--use_depth_loss
```

# 解释

depth-loss 只用来控制获取depth map
use_underwater_rasterize_formula 控制是否启用物理渲染(启用这个必须启用depth-loss)
use_depth_loss 控制是否使用depthLoss(启用这个必须启用depth-loss)



新提出的方法叫：`CG-GRR: Context-Guided Gaussian Residual Refinement`

```Shell
python leo_3dgs_trainer.py default   --data_dir /home/leo/Projects/SJTU/src/datasets/SeathruNeRF_dataset/JapaneseGradens-RedSea/undistorted_pinhole   --data_factor 2   --result_dir ./results/JapaneseGradens-RedSea-underwaterRasterizeFormula   --disable_viewer   --disable_video   --batch_size 1   --eval_steps 1000 2000 4000 6000 10000 15000 16000 22000 25000 26000 27000 28000 29000 30000   --save_steps  30000 --use_underwater_rasterize_formula --depth-loss --use_depth_loss
```

# 全量全场景
for scene in  Curasao IUI3-RedSea Panama JapaneseGradens-RedSea; do
  python leo_3dgs_trainer.py default \
    --data_dir /home/leo/Projects/SJTU/src/datasets/SeathruNeRF_dataset/${scene}/undistorted_pinhole \
    --data_factor 1 \
    --result_dir ./results/${scene}-fullMethods-factor1 \
    --disable_viewer \
    --disable_video \
    --batch_size 1 \
    --eval_steps 1000 2000 4000 6000 10000 15000 16000 22000 25000 26000 27000 28000 29000 30000 \
    --save_steps 30000 \
    --use_underwater_rasterize_formula \
    --depth-loss \
    --use_depth_loss
done


Curasao IUI3-RedSea JapaneseGradens-RedSea
# 3DGS全场景


python leo_3dgs_trainer.py default \
  --data_dir /home/leo/Projects/SJTU/src/datasets/saltpond \
  --data_factor 1 \
  --result_dir ./results/saltpond-fullMethods \
  --disable_viewer \
  --disable_video \
  --batch_size 1 \
  --eval_steps 1000 2000 4000 6000 10000 15000 16000 22000 25000 26000 27000 28000 29000 30000 \
  --save_steps 30000 \
  --gae_encoder_activation_num_steps 1000000


