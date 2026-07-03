# Factor 1

| Method                            | JapaneseGradens-RedSea PSNR ↑ | JapaneseGradens-RedSea SSIM ↑ | JapaneseGradens-RedSea LPIPS ↓ | Scene2 PSNR ↑ | Scene2 SSIM ↑ | Scene2 LPIPS ↓ | Avg PSNR ↑ | Avg SSIM ↑ | Avg LPIPS ↓ |
| --------------------------------- | -----------------------------: | -----------------------------: | ------------------------------: | -------------: | -------------: | --------------: | ----------: | ----------: | -----------: |
| 3DGS                              |                         20.970 |                         0.8592 |                           0.201 |                |                |                 |             |             |              |
| SeaSplat                          |                                |                                |                                 |                |                |                 |             |             |              |
| underwaterRasterize And depthLoss |                         22.062 |                         0.8669 |                           0.190 |                |                |                 |             |             |              |
| underwaterRasterize               |                                |                                |                                 |                |                |                 |             |             |              |

# Factor 2

| Method                            | JapaneseGradens-RedSea PSNR ↑ | JapaneseGradens-RedSea SSIM ↑ | JapaneseGradens-RedSea LPIPS ↓ | Scene2 PSNR ↑ | Scene2 SSIM ↑ | Scene2 LPIPS ↓ | Avg PSNR ↑ | Avg SSIM ↑ | Avg LPIPS ↓ |
| --------------------------------- | -----------------------------: | -----------------------------: | ------------------------------: | -------------: | -------------: | --------------: | ----------: | ----------: | -----------: |
| 3DGS                              |                         21.054 |                         0.8394 |                           0.189 |                |                |                 |             |             |              |
| SeaSplat                          |                                |                                |                                 |                |                |                 |             |             |              |
| underwaterRasterize And depthLoss |                         22.109 |                         0.8592 |                           0.157 |                |                |                 |             |             |              |
| underwaterRasterize               |                         21.094 |                         0.8509 |                           0.158 |                |                |                 |             |             |              |
| depthLoss                         |                         21.892 |                         0.8572 |                           0.160 |                |                |                 |             |             |              |
