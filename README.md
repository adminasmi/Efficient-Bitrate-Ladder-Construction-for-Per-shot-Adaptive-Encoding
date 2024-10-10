> 论文：[Efficient Bitrate Ladder Construction for Per-shot Adaptive Encoding.pdf](https://www.yuque.com/attachments/yuque/0/2024/pdf/23091796/1728543518315-0c61d83f-89f4-42ad-89e6-93d4d899b00f.pdf)
> 代码：[https://github.com/adminasmi/CAE.git](https://github.com/adminasmi/CAE.git)

---

### 构建数据集

> 要用指定编码器编码编码多个分辨率、多个码率（QP）、多个 preset（可选）。

对应的代码目录是是 `enc-dec`和 `dataset`：

+ `enc-dec` 里面是编码脚本（我目前写了 av1、vvenc 和 x265，对应的 ipynb 里有详细步骤、注释）；
+ `dataset` 里面是数据集处理（转多分辨率、划分场景）
  - `dataset`下的`analyse`主要是指标计算（PSNR, SSIM 和 VMAF，计算在 `metrics.py`里）；

> 转码时开 cpu 的 performance 模式，能快一些。

### 快速 CAE Ladder 构建

分两步：`曲线拟合`、`跨曲线预测`

> 一些和 ipynb 同名的 py 文件，基本只是把耗时的部分抽到 py 里了而已。

#### 曲线拟合

+ 对应的脚本是 `curvefit.ipynb`、`curvefit.py`，里面写了各种函数拟合的结果。
+ 结果评估在 `evalFit`文件夹里。

#### 跨曲线预测

+ 先用 `prepareData.py`生成数据集（ML 预测的数据集）；
+ 再做跨曲线参数预测：
  - `predCurve.py`是跨 preset 或跨 resolution 的参数预测；
  - `predCurveDual.py`是同时跨 preset 和 resolution 做参数预测。
+ 结果评估在 `evalPred`文件夹里。

#### 绘图

+ 一些中间结果的图基本在对应的 ipynb 里，最终的结果图和论文图都在 `draw.ipynb`里。

