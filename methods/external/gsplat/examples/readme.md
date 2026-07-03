
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

