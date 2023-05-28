import numpy as np
import spams


def normC2(X):   ##[feature, Num]
    N_feature, N_num = X.shape
    imMIN = np.mean(np.min(X, axis=0))
    imMAX = np.mean(np.max(X, axis=0))
    res = (X - imMIN) / (imMAX - imMIN +1e-6)
    divd = N_feature * 100
    return np.asfortranarray( res / divd, dtype=np.float64 ), (imMIN, imMAX, divd)


#### 1. read data
data_landsat = np.load('test_data/cia_landsat.npy')
data_modis   = np.load('test_data/cia_modis.npy')
data_landsat.shape

#### 2. specify data settings
target_index = 6

input_landsat_series  = np.delete(data_landsat, target_index, axis=0)
input_modis_series    = np.delete(data_modis,   target_index, axis=0)
input_modis_predict   = data_modis[target_index]
input_landsat_nearest = (data_landsat[target_index-1] + data_landsat[target_index+1]) / 2

#### 3. ROBOT
## 3.1 set parameters
ROBOT_beta = 1
ROBOT_lambda1 = 2
bSize = 30
step = 20
ROBOT_thresh = [0.98, 1.02]

## 3.2 spatiotemporal fusion
num_series, num_band, H, W = input_landsat_series.shape
R = int(np.ceil((H-bSize)/step)) + 1
C = int(np.ceil((W-bSize)/step)) + 1
output_predict = np.zeros_like(input_landsat_nearest)
cnt = np.zeros([H, W], dtype=np.float32) ## count overlap
for r in range(R):
    for c in range(C):
        # 1. get data
        pat_dataLD = input_landsat_series [:, :, r*step:r*step+bSize, c*step:c*step+bSize].astype(np.float32)
        pat_dataMD = input_modis_series   [:, :, r*step:r*step+bSize, c*step:c*step+bSize].astype(np.float32)
        pat_nearL  = input_landsat_nearest[:, r*step:r*step+bSize, c*step:c*step+bSize].astype(np.float32)
        pat_predM  = input_modis_predict  [:, r*step:r*step+bSize, c*step:c*step+bSize].astype(np.float32)

        ## 1. Stack the imagery
        colFDo = pat_dataLD.reshape([num_series, -1]).T
        colCDo = pat_dataMD.reshape([num_series, -1]).T
        blockH = pat_dataLD.shape[2]
        blockW = pat_dataLD.shape[3]

        colCp = pat_predM.reshape([1, -1]).T 
        colFr = pat_nearL.reshape([1, -1]).T

        ## 2. Normalization
        colFD, statsF = normC2(colFDo)
        colCD, statsC = normC2(colCDo)

        ## 3. Fusion via Optimization
        imMIN, imMAX, divd = statsC
        colCp = np.asfortranarray( (colCp - imMIN) / (imMAX - imMIN +1e-6) / divd, dtype=np.float64 )
        imMIN, imMAX, divd = statsF
        colFr = np.asfortranarray( (colFr - imMIN) / (imMAX - imMIN +1e-6) / divd, dtype=np.float64 )

        X = np.vstack([colCp, colFr*ROBOT_beta])
        D = np.vstack([colCD, colFD*ROBOT_beta])
        param = {
            "lambda1": ROBOT_lambda1,
            "numThreads": -1,
            "mode": 0,
            "pos": True,
        }
        alpha = spams.lasso(X, D, **param)
        coefRes = np.array(alpha.todense())

        ## reconstruct images using the obtained coefficients
        tmpFp = (colFDo @ coefRes).T.reshape([-1, num_band, blockH, blockW])

        resMask = ~( (ROBOT_thresh[0] < np.sum(coefRes, axis=0)) & (np.sum(coefRes, axis=0) < ROBOT_thresh[1]) )
        if np.sum(resMask) > 0: ## distributing residuals
            tmpCp = (colCDo @ coefRes).T.reshape([-1, num_band, blockH, blockW])
            tmpFp = tmpFp+ (pat_predM- tmpCp)

        ## save the results
        output_predict[:, r*step:r*step+bSize, c*step:c*step+bSize] = output_predict[:, r*step:r*step+bSize, c*step:c*step+bSize] + np.clip(tmpFp, 0, 10000)
        cnt[r*step:r*step+bSize, c*step:c*step+bSize] += 1
output_predict /= cnt

#### 4. Accuracy assessment
print("Accuracy assessment")
true_img = data_landsat[target_index] / 10000
pred_img = output_predict             / 10000
print("RMSE", np.sqrt(np.mean( (true_img - pred_img)**2 , axis=(1,2))))
print("MAE ", np.mean(np.abs(true_img - pred_img), axis=(1,2)))
print("CC  ", np.array([np.corrcoef(true_img[b].ravel(), pred_img[b].ravel())[0,1] for b in range(num_band)]))