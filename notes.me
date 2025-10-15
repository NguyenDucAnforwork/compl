- phần chuyển từ theano + downhill sang pytensor thì t làm ở file models_pytensor.py rồi nhé
- Ngoài ra cái lúc eval hàm compute_scores (file evaluation.py) chạy rất chậm -> lý do vì có vòng for lồng nhau -> t chuyển nó sang pytorch xong vectorized để tăng tốc độ chay rồi, ô thử xem có cách nào hay hơn kh
- code train chạy được r, train loss với test loss đều giảm như bình thường nhưng cái lúc eval điểm MRR gần như bằng 0, t đang hỏi chatgpt thì nó bảo lỗi từ 1 trong 3 thứ sau:
+) khi train (hàm fit trong AbstractModel của file models_pytensor), ở cái vòng for để cập nhập tham số thì trong dữ liệu của mình hiện giờ đang toàn mẫu dương -> thấy bảo như thế khiến mô hình kh học được -> đề xuất thêm negative samples vào. T thêm rồi và đang chạy test thử
+) thay MSE loss thành BCE loss (thấy bảo là loss mặc định trong paper), đổi loss xong thì train cũng chậm hơn nhiều trên máy local -> ô chạy trên gpu cho nhanh nhé
- kqua hiện giờ nó sẽ tệ như này (nãy tệ vì nó chỉ train trên 1000 mẫu thôi; t vừa fix để train full rồi, ô chạy thử nhé):
python .\fb15k_run.py
2025-10-15 19:28:32,524 (EFE)   [INFO]  Nb entities: 14951
2025-10-15 19:28:32,524 (EFE)   [INFO]  Nb relations: 1345
2025-10-15 19:28:32,524 (EFE)   [INFO]  Nb obs triples: 483142
2025-10-15 19:28:39,375 (EFE)   [INFO]  Learning rate: 0.5
2025-10-15 19:28:39,375 (EFE)   [INFO]  Max iter: 200
2025-10-15 19:28:39,376 (EFE)   [INFO]  Generated negatives ratio: 10
2025-10-15 19:28:39,376 (EFE)   [INFO]  Batch size: 4831
2025-10-15 19:28:39,376 (EFE)   [INFO]  Starting grid search on: DistMult_Model
Training DistMult_Model with PyTensor...
Model initialized with 3259200 parameters
Dimensions: n=14951, m=1345, l=14951, k=200
C:\Users\ADMIN\anaconda3\Lib\site-packages\pytensor\link\c\cmodule.py:2968: UserWarning: PyTensor could not link to a BLAS installation. Operations that might benefit from BLAS will be severely degraded.
This usually happens when PyTensor is installed via pip. We recommend it be installed via conda/mamba/pixi instead.
Alternatively, you can use an experimental backend such as Numba or JAX that perform their own BLAS optimizations, by setting `pytensor.config.mode == 'NUMBA'` or passing `mode='NUMBA'` when compiling a PyTensor function.
For more options and details see https://pytensor.readthedocs.io/en/latest/troubleshooting.html#how-do-i-configure-test-my-blas-library
  warnings.warn(
***INITIAL VALUES: *** [-1.6722885  -0.40823883  0.3435664   0.63266075 -2.676425   -1.1792037
 -2.3668866   0.24606825  1.8430544  -1.1888553   0.96416914 -0.9623589
  0.8516909  -0.33505395  0.2996708   0.57539284 -0.12150055 -0.9373728
 -0.5039388   0.69794834  0.24648361 -0.5722082   1.0582381  -0.9191683
 -2.875467   -1.576963   -0.20441653 -0.677291    0.26676363  0.24731821
  0.32862252 -0.24057633 -1.3140643  -0.05883687  0.34997123 -0.26542422
 -0.65705216 -1.0102142   0.97999436  0.41322303 -0.19622368 -0.6769401
 -1.760295    0.07898924  0.7857605  -0.7062548  -0.5943624   0.19108461
  0.9342251  -0.58954376  1.7425348   0.40794215 -0.17713219 -0.49959812
  0.3988874   1.3110638   0.5398412   0.23636189 -1.1908455  -1.3711407
  0.19472557  1.9496994  -0.94076663  1.4450893  -0.12883954  1.2960919
  0.9606697   0.7305727  -1.5079008   1.2778293   0.952711   -0.8022314
 -0.64526594 -0.81910616  0.35015476 -0.04355703 -0.4368359   0.46074083
  1.4502488   1.3135111  -0.2196218   1.1247365   0.62180024 -0.72299033
 -0.9901644  -0.06441387  0.21738549  0.27453238 -0.80299175 -0.70518905
 -1.0188571  -1.1219856  -0.23000641  0.14672421 -0.8739881  -0.28831285
  0.22199017 -1.0600268  -1.2901025   0.14820224 -0.9042383   0.4628779
  1.259403   -0.24427572 -0.6618986  -2.1878505  -0.396624   -0.17411439
 -1.7733183  -0.7953662   0.7535931   2.089165    1.1209549  -0.5158145
 -0.78934133 -0.05364887 -0.25715682  2.034082   -0.50776076  0.07420436
 -1.2528194  -0.48051435  0.34840336 -1.0083146  -0.14533009 -0.39421752
 -0.811033   -0.9805439   0.5323925   1.3186172   1.2469432   0.65099204
 -0.6801629   0.21306725 -0.35931832 -1.1174338  -0.21899886  0.29836604
 -1.2542329   1.919384   -0.61358935  0.7671858   0.15497555  1.6350338
 -0.5888209  -2.0791018  -0.888226    0.5456578   0.459532    1.8659202
  2.982809    1.8638645   0.43026462  0.01051286 -0.5305144  -1.0871955
 -0.828909   -0.6042251  -0.3641517  -1.2457627   1.0009984   0.2835264
 -0.9592234  -1.3215091   1.2445441   0.01757956 -0.6752736  -0.2538183
 -0.03639517 -0.9483044  -0.50026053  0.48877347  2.0098991   0.83097124
 -0.5032242   1.2315356   0.71353304  0.20412165  1.661694    0.0086128
 -0.66502684 -1.5916452   0.7833916  -1.0329686   0.3160839   0.15111256
 -0.40377903  1.7936432  -1.6770462  -1.1366781  -0.08186491  0.7804608
 -1.0813187  -1.1534389   1.0521439   0.66256696 -2.1157691  -1.3370757
 -1.3630275  -0.6201445 ]
Epoch 0: Loss = 5.6827
Epoch 100: Loss = 5.6949
Test loss: 5.7161
2025-10-15 19:31:09,809 (EFE)   [INFO]  Grid search finished
2025-10-15 19:31:09,809 (EFE)   [INFO]  Validation metrics:
2025-10-15 19:31:09,809 (EFE)   [INFO]  Model                   MRR     RMRR    H@1     H@3     H@10    rank    lmbda
2025-10-15 19:31:09,810 (EFE)   [INFO]  DistMult_Model  0.001   0.001   0.000   0.000   0.001   200     0.010000
2025-10-15 19:31:09,811 (EFE)   [INFO]  Corresponding Test metrics:
2025-10-15 19:31:09,811 (EFE)   [INFO]  DistMult_Model  0.001   0.001   0.000   0.000   0.001   200     0.010000
