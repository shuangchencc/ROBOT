import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def ROBOT_harmoniz(dataLD, clrLD, dataMD, Ddates=None, dataM=None, Mdates=None, 
                   indexL=np.array([0,1,2,3,4,5]), indexM=np.array([0,1,2,3,4,5]),
                   harmoniz_val_thresh=np.array([0.9, 1.1]), harmoniz_decayRatio=0.73, 
                   harmoniz_minPatchSize=200, harmoniz_series_minVal=10000, harmoniz_pair_minVal=1000, harmoniz_validRange=np.array([0.8, 1.2])):
    ## dataLD: Landsat images as an array, [Num images, Num bands, weight, height]
    ## clrLD:  clear-sky masks of Landsat images as an array, True indicates clear-sky and False indicates missing or contaminated, [Num images, weight, height]
    ## dataMD: MODIS images as an array, [Num images, Num bands, weight, height]
    ## Ddates: julian acquisition dates (str) of dataLD and dataMD as a Python list, e.g. ["2000001", "2000365". ...]
    ## dataM: MODIS data at reconstruction temporal phases, [Num images (to be reconstructed), Num bands, weight, height]
    ## Mdates: julian acquisition dates (str) of dataM as a Python list, e.g. ["2000001", "2000365". ...]

    assert len(indexL) == len(indexM)
    Nd, Bf, H, W = dataLD.shape
    Nd, Bc, H, W = dataMD.shape
    
    ############################ get the mask of valid pixels
    valLD = np.zeros_like(clrLD)
    for i in prange(Nd):
        ratios   = np.sum(dataLD[i][indexL] / (dataMD[i][indexM] + 1e-6), axis=0) / Bf
        valLD[i] = clrLD[i] & (harmoniz_val_thresh[0] < ratios) & (ratios < harmoniz_val_thresh[1])

    ############################ time-series harmonization
    bSize = max(H, W)
    decayRatio = harmoniz_decayRatio
    numIters = int(np.ceil(np.log(harmoniz_minPatchSize / max(H, W)) / np.log(decayRatio)))
    coefM = np.ones((Bc, H, W), dtype=np.float32)
    for _ in range(numIters):
        R = int(np.ceil((H-bSize)/bSize)) + 1
        C = int(np.ceil((W-bSize)/bSize)) + 1
        for ith in prange(R*C):
            r = ith // C; c = ith % C
            pat_valLD  = valLD [:,    r*bSize:r*bSize+bSize, c*bSize:c*bSize+bSize]
            pat_dataLD = dataLD[:, :, r*bSize:r*bSize+bSize, c*bSize:c*bSize+bSize]
            pat_dataMD = dataMD[:, :, r*bSize:r*bSize+bSize, c*bSize:c*bSize+bSize]
            N_pixels = np.sum(pat_valLD)
            ## skip, if there are not enough valid pixels
            if N_pixels < harmoniz_series_minVal:
                continue
            else:
                for idx in range(len(indexL)):
                    coef = np.linalg.lstsq(pat_dataMD[:, indexM[idx]].ravel()[pat_valLD.ravel()][:, np.newaxis].astype(np.float32), 
                                           pat_dataLD[:, indexL[idx]].ravel()[pat_valLD.ravel()].astype(np.float32))[0][0]
                    if harmoniz_validRange[0] < coef < harmoniz_validRange[1]:
                        if dataM is not None:
                            coefM[indexM[idx],r*bSize:r*bSize+bSize, c*bSize:c*bSize+bSize] = coefM[indexM[idx],r*bSize:r*bSize+bSize, c*bSize:c*bSize+bSize] * coef
                        pat_dataMD[:, indexM[idx]] = pat_dataMD[:, indexM[idx]] * coef
        bSize = int(np.ceil(bSize*decayRatio))
        
    ############################ image patch pairs harmonization
    ## smoothing the coeficients time series may help improve data quality...
    coefM_Nd = np.ones((Nd, Bc, H, W), dtype=np.float32)
    for idx_img in range(Nd):
        bSize = max(H, W)
        for _ in range(numIters):
            R = int(np.ceil((H-bSize)/bSize)) + 1
            C = int(np.ceil((W-bSize)/bSize)) + 1
            for r in prange(R):
                for c in range(C):
                    pat_valLD  = valLD [idx_img,    r*bSize:r*bSize+bSize, c*bSize:c*bSize+bSize]
                    pat_dataLD = dataLD[idx_img, :, r*bSize:r*bSize+bSize, c*bSize:c*bSize+bSize]
                    pat_dataMD = dataMD[idx_img, :, r*bSize:r*bSize+bSize, c*bSize:c*bSize+bSize]
                    N_pixels = np.sum(pat_valLD)
                    ## skip, if there are not enough valid pixels
                    if N_pixels < harmoniz_pair_minVal:
                        continue
                    else:
                        for idx in range(len(indexL)):
                            coef = np.linalg.lstsq(pat_dataMD[indexM[idx]].ravel()[pat_valLD.ravel()][:, np.newaxis].astype(np.float32), 
                                                   pat_dataLD[indexL[idx]].ravel()[pat_valLD.ravel()].astype(np.float32))[0][0]
                            if harmoniz_validRange[0] < coef < harmoniz_validRange[1]:
                                if dataM is not None:
                                    coefM_Nd[idx_img, indexM[idx], r*bSize:r*bSize+bSize, c*bSize:c*bSize+bSize] = coefM_Nd[idx_img, indexM[idx], r*bSize:r*bSize+bSize, c*bSize:c*bSize+bSize] * coef
                                pat_dataMD[indexM[idx]] = pat_dataMD[indexM[idx]] * coef
            bSize = int(np.ceil(bSize*decayRatio))
            
    if dataM is not None:
        N, Bc, H, W = dataM.shape
        ## harmonizing dataM
        for i in prange(N):
            tgtDate = Mdates[i]
            if tgtDate in Ddates:
                tgtIDX = np.where(Ddates == tgtDate)[0][0]
                coefM_this = coefM * coefM_Nd[tgtIDX]
            else:
                leftDate, rightDate = -9999, -9999
                for date in Ddates:
                    if ((leftDate != -9999) and (leftDate < date < tgtDate)) or\
                        ((leftDate == -9999) and (date < tgtDate)):
                        leftDate = date
                    if ((rightDate != -9999) and (tgtDate < date < rightDate)) or\
                        ((rightDate == -9999) and (tgtDate < date)):
                        rightDate = date
                if leftDate == -9999:
                    idxR = np.where(Ddates == rightDate)[0][0]
                    coefM_this = coefM * coefM_Nd[idxR]
                elif rightDate == -9999:
                    idxL = np.where(Ddates == leftDate)[0][0]
                    coefM_this = coefM * coefM_Nd[idxL]
                else:
                    idxL = np.where(Ddates == leftDate)[0][0]
                    idxR = np.where(Ddates == rightDate)[0][0]
                    
                    # dif1 = (np.sum(np.power(dataM[i].astype(np.float32) - dataMD[idxL], 2), axis=0) / 10000.  +1e-6).astype(np.float32)
                    # dif2 = (np.sum(np.power(dataM[i].astype(np.float32) - dataMD[idxR], 2), axis=0) / 10000.  +1e-6).astype(np.float32)
                    dif1 = (np.power(dataM[i].astype(np.float32) - dataMD[idxL], 2) / 10000.  +1e-6).astype(np.float32)
                    dif2 = (np.power(dataM[i].astype(np.float32) - dataMD[idxR], 2) / 10000.  +1e-6).astype(np.float32)
                    dif1[:] = np.sum(dif1, axis=0)
                    dif2[:] = np.sum(dif2, axis=0)
                    
                    denominator = dif1 + dif2
                    coefM_this = coefM * ((dif2 / denominator) * coefM_Nd[idxL] + (dif1 / denominator) * coefM_Nd[idxR])
            dataM[i] = dataM[i] * coefM_this
            
        ## smoothing dataM
        win_size = 2
        for ith in prange(H*W):
            idxR = ith // W; idxC = ith % W
            for idxB in range(Bc):
                dataM[win_size:-win_size, idxB, idxR, idxC] = np.convolve(dataM[:, idxB, idxR, idxC], [1/(2*win_size+1) for i in range(2*win_size+1)])[2*win_size:-2*win_size]
            
    return dataMD, dataM
