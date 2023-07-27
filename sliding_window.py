# -*- coding: utf-8 -*-
import decimal


import numpy as np
import pandas as pd

from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew, mode


# Raise exception for warnings
import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from log import log
else:
    # uses current package visibility
    from .log import log

def round_half_up(x):
    ''' round half up instead of to nearest even integer '''
    return int(decimal.Decimal(x).to_integral_value(rounding=decimal.ROUND_HALF_UP))

# -------------------
#  Feature Extraction
# -------------------

def zcc(data,*args,**kwargs):
    ''' Zero crossing count '''
    return np.nonzero(np.diff(data > 0))[0].shape[0]

def mcc(data,*args,**kwargs):
    ''' Mean crossing count '''
    return np.nonzero(np.diff((data-np.mean(data)) > 0))[0].shape[0]

def energy(data,*args,**kwargs):
    ''' Energy of a matrix A*A.T '''
    return np.dot(data.T,data)

def autocorr(data, min_delay=10):
    ''' Statistical correlation with a lag of t '''
    result = np.correlate(data, data,mode='full')
    result = result[min_delay+result.shape[0]//2:]
    idx = np.argmax(result)
    return np.array([result[idx], idx])

def IQR(data,q=[75,25]):
    ''' Interquartile range '''
    return np.subtract(*np.percentile(data, q))

def FFT_peaks(data,*args,**kwargs):
    ''' Highest FFT Value and Frequency + Ratio between highest and second highest peak '''
    locs,props = find_peaks(data,height=(None, None))
    pks = props['peak_heights']
    
    idx = pks.argsort()[::-1]
    pks,locs = pks[idx],locs[idx]

    if pks.shape[0] == 0:
        pks,locs = data,np.array([0])
    
    elif pks.shape[0] == 1:
        pks = np.concatenate((pks,[np.min(data)]))
    
    finterval = 50/(data.shape[0]-1)
    return np.array([pks[0],locs[0]*finterval,pks[0]/(pks[1]+np.finfo(float).eps)])

def SlideEnergy(x,x2,winlen,skiplen,fs2,finterval):
    nwin = np.fix((fs2-(winlen-skiplen))/skiplen).astype(int)
    y = np.zeros(nwin)
    for n in range(nwin):
        idx_start = round_half_up( (n)*skiplen/finterval )
        idx_end = min((round_half_up( ((n)*skiplen + winlen)/finterval  ) + 1, x.shape[0]))
        y[n] = np.sum(x2[idx_start:idx_end])
    return np.array(y)


def FFT_subbands(data,*args,**kwargs):
    fs2 = 50
    finterval = fs2/(data.shape[0]-1)
    data2 = np.power(data,2)
    data2_sum = data2.sum()
    
    result = np.empty(846)
    idx = 0
    for winlen in [1,2,3,4,5,10,15,20,25]:
        y = SlideEnergy(data,data2,winlen,.5 if winlen==1 else 1.,fs2,finterval)
        result[idx:idx+y.shape[0]] = y
        result[idx+y.shape[0]:idx+2*y.shape[0]] = y/data2_sum
        idx+=2*y.shape[0]
    return result
    
# -------------------
#  List of Features
# -------------------

def val_FX_list(FX_list):
    if not isinstance(FX_list,list):
        return [str(FX_list)]
    else:
        return [x for x in FX_list if x in FXdict]

def get_FX_list(P):
    return val_FX_list(FEATURES[P.get('FX_sel')])

def get_FX_list_len(FX_list):
    return sum(num for _,_,num,_ in [FXdict[Fx] for Fx in FX_list])

def get_FX_names(indeces=None):
    names = []
    for name in [*FXdict]:
        if name == 'auto_correlation':
            names += ['auto_correlation_peak','auto_correlation_peak_idx']
        elif name == 'Peak_fft':
            names += ['peak_fft','peak_fft_fq','peak_fft_ratio']
        elif name == 'Subband':
            for bandwith in [1,2,3,4,5,10,15,20,25]:
                stepsize = .5 if bandwith==1 else 1.
                center_fqs = np.arange(start=(bandwith/2),stop=(stepsize+50-bandwith/2),step=stepsize)

                for center_fq in center_fqs:
                    names.append(f'bw_{bandwith}_cfq_{center_fq:.1f}')
                    
                for center_fq in center_fqs:
                    names.append(f'bw_{bandwith}_cfq_{center_fq:.1f}_ratio')
        else:
            names.append(name)
    
    names = np.array(names)
    
    if indeces is None:
        return names
    else:
        return names[indeces]


''' name: (func, params, number of return values, fft) '''
FXdict = {
  "mean": (np.mean,None,1,False),
  "std": (np.std,None,1,False),
  #"zcr": (zcc,None,1,False),
  "mcr": (mcc,None,1,False),
  "energy": (energy,None,1,False),
  "auto_correlation": (autocorr,1,2,False),
  "kurtosis": (kurtosis,None,1,False),
  "skew": (skew,None,1,False),
  
  "mean_fft": (np.mean,None,1,True),
  "std_fft": (np.std,None,1,True),
  "energy_fft": (energy,None,1,True),
  "kurtosis_fft": (kurtosis,None,1,True),
  "skew_fft": (skew,None,1,True),
  "DC_fft": (lambda data,_:data[0],None,1,True),
  "Peak_fft": (FFT_peaks,None,3,True),
  "Subband": (FFT_subbands,None,846,True),
}

QUARTILES = [0,5,10,25,50,75,90,95,100]

# Append quartiles
for q in QUARTILES:
    FXdict["Q%d"%q] = (np.percentile,q,1,False)

# Append quartile ranges
for i in range(len(QUARTILES)-1):
    for j in range(i+1,len(QUARTILES)):
        FXdict["IQR_%d_%d"%(QUARTILES[i], QUARTILES[j])] = (IQR,[QUARTILES[j], QUARTILES[i]],1,False)
    

FEATURES = {
    'basic': ['mean','median','std','mcr','kurtosis','skew'],
    'all': [*FXdict],
    'auto_correlation': ['auto_correlation'],
    }

# Feature indeces sorted by mRMR on Preview
#FEATURE_INDECES = [864, 271, 123, 276, 9, 552, 881, 638, 883, 459, 898, 372, 130, 887, 551, 19, 553, 863, 884, 267, 129, 865, 467, 885, 460, 1, 364, 117, 876, 880, 122, 367, 877, 140, 878, 270, 889, 215, 121, 560, 890, 461, 891, 365, 874, 373, 14, 458, 882, 886, 366, 875, 141, 879, 275, 554, 896, 899, 895, 139, 894, 468, 892, 268, 900, 462, 888, 124, 897, 561, 870, 647, 901, 363, 866, 128, 8, 869, 724, 871, 868, 142, 131, 639, 903, 873, 269, 904, 648, 902, 559, 905, 723, 906, 715, 893, 277, 466, 313, 127, 20, 907, 179, 125, 216, 374, 791, 503, 410, 646, 595, 368, 504, 725, 138, 596, 469, 409, 314, 782, 790, 677, 180, 872, 678, 562, 222, 371, 749, 848, 320, 750, 143, 811, 637, 415, 812, 319, 126, 296, 10, 847, 30, 722, 416, 509, 789, 508, 792, 33, 555, 414, 649, 223, 28, 31, 558, 506, 0, 192, 839, 13, 645, 221, 218, 172, 412, 132, 316, 181, 4, 144, 29, 846, 597, 598, 726, 321, 716, 32, 137, 507, 171, 505, 849, 318, 3, 510, 315, 411, 188, 264, 679, 278, 266, 751, 680, 34, 392, 220, 813, 193, 26, 600, 752, 375, 636, 413, 21, 6, 599, 465, 601, 7, 274, 814, 397, 35, 417, 788, 217, 295, 602, 793, 511, 682, 36, 683, 18, 681, 224, 182, 317, 15, 754, 145, 721, 755, 753, 291, 684, 867, 816, 27, 470, 815, 118, 817, 301, 756, 845, 783, 23, 197, 818, 603, 219, 640, 37, 650, 198, 685, 322, 300, 644, 418, 651, 512, 850, 757, 819, 116, 563, 189, 24, 727, 22, 487, 604, 387, 25, 305, 38, 569, 17, 686, 393, 206, 758, 419, 323, 463, 513, 199, 820, 225, 652, 170, 477, 840, 605, 146, 687, 178, 44, 42, 570, 759, 279, 420, 821, 191, 324, 200, 325, 228, 164, 43, 309, 284, 361, 227, 226, 272, 155, 514, 45, 653, 606, 290, 380, 370, 190, 607, 51, 515, 688, 794, 516, 421, 406, 232, 173, 52, 728, 329, 501, 760, 231, 422, 608, 787, 50, 156, 425, 689, 383, 822, 594, 330, 326, 519, 517, 717, 426, 654, 423, 520, 41, 475, 53, 183, 761, 213, 474, 690, 287, 609, 823, 154, 39, 427, 376, 762, 381, 424, 207, 327, 233, 557, 297, 331, 611, 407, 518, 478, 521, 564, 824, 691, 288, 567, 328, 502, 610, 612, 844, 763, 205, 229, 568, 54, 169, 332, 693, 825, 49, 310, 235, 476, 428, 655, 692, 208, 333, 765, 234, 581, 694, 593, 764, 384, 522, 827, 157, 311, 613, 283, 826, 408, 46, 566, 57, 766, 147, 429, 656, 55, 212, 59, 720, 828, 163, 56, 230, 590, 58, 136, 236, 312, 695, 714, 523, 60, 729, 500, 767, 265, 614, 829, 334, 471, 571, 696, 405, 851, 615, 61, 430, 158, 402, 524, 768, 214, 830, 209, 697, 237, 385, 769, 616, 308, 831, 580, 431, 285, 335, 165, 62, 525, 40, 583, 133, 698, 210, 770, 47, 832, 582, 617, 382, 591, 479, 432, 699, 336, 771, 68, 730, 833, 337, 795, 526, 643, 48, 781, 239, 618, 63, 433, 120, 527, 244, 497, 341, 700, 657, 619, 177, 238, 772, 159, 64, 834, 498, 529, 528, 701, 620, 240, 592, 773, 166, 835, 434, 435, 437, 550, 338, 243, 69, 77, 491, 74, 148, 702, 379, 621, 342, 774, 836, 488, 75, 731, 436, 339, 703, 531, 565, 775, 530, 241, 532, 195, 623, 705, 67, 438, 777, 533, 72, 499, 85, 784, 340, 622, 211, 776, 480, 704, 248, 66, 73, 65, 168, 624, 245, 706, 778, 472, 439, 489, 76, 345, 282, 625, 707, 779, 2, 343, 440, 732, 242, 852, 709, 346, 84, 628, 572, 708, 710, 441, 196, 627, 626, 534, 442, 78, 536, 535, 796, 82, 711, 629, 344, 632, 247, 448, 83, 537, 631, 249, 633, 541, 404, 712, 540, 634, 443, 353, 306, 542, 96, 70, 347, 664, 71, 80, 630, 490, 539, 352, 473, 447, 260, 446, 663, 357, 445, 256, 350, 449, 194, 733, 544, 543, 545, 635, 109, 94, 665, 351, 786, 453, 538, 254, 246, 358, 79, 450, 95, 255, 354, 349, 744, 253, 98, 289, 547, 86, 403, 452, 444, 451, 261, 546, 252, 259, 454, 745, 107, 355, 348, 806, 251, 548, 99, 573, 359, 97, 662, 90, 101, 81, 106, 257, 356, 92, 110, 105, 807, 455, 250, 797, 88, 91, 104, 100, 93, 89, 280, 108, 258, 102, 808, 111, 360, 378, 87, 262, 112, 263, 103, 114, 113, 746, 853, 658, 184, 160, 115, 734, 843, 809, 798, 810, 286, 747, 800, 748, 837, 841, 396, 377, 799, 672, 204, 854, 740, 739, 386, 741, 456, 668, 395, 673, 394, 855, 667, 801, 674, 856, 577, 669, 203, 742, 578, 307, 464, 675, 676, 556, 575, 576, 857, 713, 743, 574, 135, 659, 579, 12, 457, 201, 481, 670, 369, 202, 153, 735, 149, 134, 486, 273, 641, 719, 671, 660, 858, 482, 185, 483, 150, 151, 299, 281, 303, 484, 838, 485, 152, 304, 167, 162, 161, 802, 666, 780, 785, 302, 642, 362, 585, 298, 586, 842, 388, 587, 736, 174, 391, 862, 175, 389, 176, 859, 492, 737, 390, 803, 718, 5, 588, 16, 804, 549, 493, 494, 661, 400, 401, 584, 805, 589, 399, 495, 186, 292, 738, 187, 11, 860, 861, 496, 398, 294, 119, 293]   

# Feature indeces sorted by mRMR on User 1
FEATURE_INDECES = [887, 198, 553, 881, 117, 19, 864, 882, 888, 907, 865, 638, 215, 875, 863, 461, 906, 874, 879, 886, 892, 554, 876, 883, 904, 889, 897, 871, 367, 893, 901, 880, 9, 877, 373, 715, 1, 878, 462, 884, 885, 8, 890, 366, 894, 891, 271, 868, 468, 782, 873, 896, 14, 895, 131, 839, 870, 869, 276, 898, 460, 900, 905, 0, 132, 13, 867, 467, 899, 872, 903, 555, 116, 866, 264, 902, 270, 140, 320, 222, 7, 223, 372, 416, 129, 4, 33, 415, 560, 321, 3, 269, 34, 319, 368, 509, 141, 510, 595, 596, 130, 503, 410, 598, 118, 218, 677, 142, 409, 365, 504, 678, 314, 139, 417, 749, 316, 508, 637, 750, 32, 811, 551, 812, 511, 275, 313, 10, 506, 216, 680, 463, 414, 561, 752, 128, 814, 597, 272, 601, 224, 412, 507, 599, 602, 600, 23, 679, 36, 841, 751, 683, 813, 640, 20, 277, 684, 682, 315, 755, 559, 681, 361, 221, 817, 505, 411, 838, 754, 753, 31, 24, 35, 18, 756, 784, 816, 815, 818, 466, 217, 781, 28, 413, 603, 318, 37, 717, 322, 17, 418, 685, 757, 819, 512, 123, 219, 143, 21, 317, 220, 456, 38, 323, 30, 225, 419, 647, 26, 127, 604, 458, 25, 686, 641, 22, 758, 820, 133, 138, 29, 552, 27, 369, 513, 126, 605, 759, 549, 821, 687, 842, 427, 268, 235, 371, 521, 324, 331, 332, 11, 695, 613, 718, 693, 767, 765, 785, 694, 226, 827, 764, 614, 840, 692, 428, 233, 696, 829, 826, 766, 522, 53, 822, 611, 615, 760, 639, 828, 612, 520, 768, 333, 125, 697, 610, 121, 429, 688, 234, 54, 42, 830, 519, 769, 606, 374, 823, 556, 529, 227, 616, 761, 523, 435, 783, 59, 52, 831, 698, 57, 530, 426, 58, 436, 689, 518, 339, 437, 646, 341, 824, 607, 618, 425, 770, 424, 340, 762, 617, 243, 420, 528, 619, 329, 342, 527, 514, 690, 700, 825, 43, 531, 832, 334, 699, 763, 244, 621, 241, 716, 620, 524, 434, 526, 12, 242, 330, 430, 608, 771, 701, 691, 236, 772, 433, 328, 432, 56, 438, 74, 265, 702, 41, 622, 75, 245, 73, 703, 55, 833, 232, 532, 68, 515, 773, 609, 69, 525, 337, 338, 834, 70, 623, 39, 230, 76, 336, 774, 704, 421, 77, 835, 78, 343, 231, 64, 431, 439, 775, 72, 325, 642, 63, 71, 459, 533, 240, 624, 238, 705, 836, 516, 79, 335, 239, 62, 237, 776, 363, 344, 625, 440, 422, 60, 706, 534, 16, 246, 517, 719, 777, 49, 843, 626, 707, 778, 47, 295, 441, 535, 65, 345, 326, 536, 442, 469, 346, 80, 48, 627, 423, 247, 137, 248, 178, 708, 786, 443, 779, 61, 537, 347, 67, 628, 177, 228, 249, 83, 84, 444, 538, 348, 169, 709, 629, 327, 558, 539, 445, 392, 250, 40, 349, 362, 51, 85, 82, 251, 81, 446, 2, 710, 350, 540, 252, 648, 630, 50, 87, 66, 86, 351, 724, 447, 179, 711, 91, 88, 457, 90, 89, 541, 253, 631, 254, 352, 92, 46, 662, 712, 448, 632, 542, 93, 255, 229, 96, 663, 353, 274, 487, 94, 97, 95, 291, 449, 633, 387, 543, 580, 98, 634, 256, 354, 44, 99, 450, 576, 635, 544, 180, 257, 723, 355, 173, 451, 545, 739, 124, 100, 101, 296, 102, 452, 664, 356, 546, 258, 290, 103, 547, 453, 357, 170, 454, 575, 548, 259, 577, 358, 455, 359, 45, 260, 740, 360, 261, 107, 661, 104, 262, 488, 108, 105, 464, 263, 578, 266, 393, 106, 113, 109, 738, 112, 114, 483, 581, 6, 110, 665, 111, 115, 482, 579, 485, 486, 562, 484, 806, 181, 741, 667, 666, 172, 388, 394, 805, 391, 645, 182, 489, 582, 168, 144, 807, 297, 481, 390, 171, 742, 490, 395, 737, 298, 389, 668, 574, 583, 396, 183, 299, 804, 465, 837, 736, 660, 743, 744, 862, 803, 186, 187, 808, 386, 491, 294, 802, 735, 300, 184, 185, 791, 669, 859, 584, 659, 188, 397, 860, 861, 292, 189, 809, 293, 790, 190, 801, 745, 714, 658, 492, 120, 858, 670, 734, 301, 364, 725, 810, 585, 800, 857, 167, 191, 134, 573, 398, 289, 671, 780, 733, 746, 856, 586, 493, 799, 855, 192, 722, 278, 798, 844, 672, 302, 848, 657, 494, 480, 587, 854, 732, 399, 572, 747, 193, 787, 847, 797, 176, 731, 495, 400, 305, 174, 199, 673, 720, 194, 401, 588, 304, 656, 748, 303, 643, 496, 550, 195, 200, 402, 375, 571, 589, 853, 145, 497, 730, 161, 590, 286, 175, 160, 479, 306, 196, 674, 796, 655, 385, 498, 403, 197, 792, 15, 789, 166, 308, 273, 654, 162, 404, 713, 591, 649, 203, 204, 478, 206, 307, 201, 675, 202, 729, 205, 207, 499, 852, 676, 470, 122, 636, 405, 383, 152, 653, 384, 382, 570, 849, 557, 795, 476, 846, 592, 309, 567, 500, 568, 477, 285, 288, 569, 379, 146, 282, 381, 593, 287, 151, 406, 153, 475, 165, 208, 281, 214, 563, 136, 473, 267, 594, 474, 728, 378, 652, 501, 156, 279, 566, 159, 726, 283, 502, 310, 851, 312, 119, 407, 370, 408, 163, 380, 155, 213, 154, 157, 284, 209, 565, 794, 150, 164, 311, 210, 472, 845, 212, 376, 377, 158, 651, 135, 211, 793, 788, 148, 727, 650, 149, 147, 644, 850, 471, 280, 721, 564, 5]

def get_best_n_features(n):return FEATURE_INDECES[:min(n,len(FEATURE_INDECES))]

# -------------------
#  Padding
# -------------------

def zerol(data,winsize):
    return np.concatenate((np.zeros((data.shape[0],winsize-1)),data),axis=1)

def zeror(data,winsize):
    return np.concatenate((data,np.zeros((data.shape[0],winsize-1))),axis=1)
    
def mirrorl(data,winsize):
    return np.concatenate((data[:winsize-1:0],data),axis=1)
    
def mirrorr(data,winsize):
    return np.concatenate((data,data[:-winsize-1:-1]),axis=1)

def default(data,winsize):
    log("This padding mode is unknown, zerol is applied",error=True)
    return zerol(data,winsize)

def make_numpy(mat):
    if not isinstance(mat, np.ndarray):
        if isinstance(mat,list):
            return np.array(mat)
        elif isinstance(mat, pd.DataFrame):
            return mat.to_numpy()
        else:
            log("Unknown data type: "+str(type(mat)),error=True)
    return mat

def slidingWindow(P,data,label=None):
    winsize = P.get('winsize')
    jumpsize = P.get('jumpsize')
    FX_list = get_FX_list(P)
    padding = P.get('padding')
    
    if not np.isscalar(winsize) or winsize < 1 or int(winsize) != winsize:
        log("slidingWindow: winsize must be integer and larger or equal to 1",error=True)
        return None
    
    if not np.isscalar(jumpsize) or jumpsize < 1 or int(jumpsize) != jumpsize:
        log("slidingWindow: jumpsize must be integer and larger or equal to 1",error=True)
        return None
    
    data = make_numpy(data)
    
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    
    if data.ndim != 2:
        log("slidingWindow: data must be two-dimensional matrix. data.ndim = %d"%data.ndim,error=True)
        return None
    
    colNo = data.shape[0]
    rowNo = data.shape[1]
    
    if colNo == rowNo:
        log("slidingWindow: data must be a matrix with one dimension (along which the window is sliding) longer than the other one",error=True)
        return None
    
    if colNo > rowNo:
        data = np.transpose(data)

    sdata = data.shape[1]                   # size of the time series
    s = np.ceil(sdata/jumpsize).astype(int) # size of output timeline after sliding window
    FXnr = get_FX_list_len(FX_list)                 # number of output features
    Cnr = data.shape[0]                     # number of output channels
    
    # Create the output data
    X = np.empty((s,Cnr,FXnr))
    if label is not None:
        label = make_numpy(label)
        Y = np.empty((s))

    ## Pad the data
    # There are several padding modes:
    # In a sliding window process, the first sliding window of size winsize 
    # could reach to element outside (on the left) of the vector. Similarly
    # the last sliding window could reach to elements outside (on the right) of
    # the end of the vector.
    # Several padding strategies are available to ensure the output vector is
    # of same size as the input:
    # 
    # We pad the vector with null at the front to ensure fast loops later.
    switcher = {
        'zerol': zerol,
        'zeror': zeror,
        'mirrorl': mirrorl,
        'mirrorr': mirrorr
    }
    func = switcher.get(padding,default)
    
    data = func(data,winsize)
    
    # Iterate all the windows
    wi=0    # wi index in the output (window index)
    for w in range(0,sdata,jumpsize):
        
        # Iterate all channels
        for j in range(0,Cnr):
            # Extract the windowed data
            data_win = data[j,w:w+winsize]
            
            # Calculate FFT
            data_fft = fft(data_win)
            data_fft = abs(data_fft[:data_fft.shape[0]//2+1])
            DC_fft = data_fft[0]
            
            # Silencing 0-0.5Hz
            finterval = 50/data_fft.shape[0]
            idxp5 = np.fix(0.5/finterval).astype(int)+1
            data_fft[:idxp5] = 0 
            
            # Calculate all features
            i=0
            for FX_name in FX_list:
                # Load Feature function
                FX, FXOpts, L, rq_fft = FXdict[FX_name]
                
                # Compute the feature
                if FX_name == 'DC_fft':
                    fx = DC_fft
                elif rq_fft:
                    fx = FX(data_fft,FXOpts)
                else:
                    fx = FX(data_win,FXOpts)

                # Update the output vector
                X[wi,j,i:i+L] = fx
                
                i+=L
                
        if label is not None:
            Y[wi] = mode(label[w:w+winsize])[0]       
            
        # increase output index
        wi += 1

    
    if label is not None:
        return X.squeeze(), Y
    else:
        return X.squeeze()
    
if __name__ == "__main__":
    data = np.array([0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,0,1,1,1,1,1,2,3,4,5,])
    data_fft = fft(data)
    data_fft = np.abs(data_fft[:data_fft.shape[0]//2+1]).round(13)
    fs2 = 50
    finterval = fs2/(data_fft.shape[0]-1)
    idxp5 = np.fix(0.5/finterval).astype(int)+1
    data_fft[:idxp5] = 0 
    
    print("Data:",data)
    i=0
    for name, (FX, FXOpts, num, rq_fft) in FXdict.items():
        i+=num
        if rq_fft:
            fx = FX(data_fft,FXOpts)
        else:
            fx = FX(data,FXOpts)
        print(i,name+':',fx,num)


    from params import Params
    
    P = Params(FX_sel = 'all',winsize=5,jumpsize=5)
    #print(slidingWindow(P,data,label=None))
    
    # print(data_fft)
    #slide = SlideEnergy(data_fft,np.power(data_fft,2),1,0.5,fs2,finterval)
    #slide = SlideEnergy(data_fft,np.power(data_fft,2),25,1,fs2,finterval)
    
    # print(data_fft.shape[0])

    # print(slide)
    # print(slide.shape)
    # print(slide[80-1])

    # print(np.power(data_fft,2)[10:11])
    
    #print(get_FX_names(np.random.choice(a=[False, True], size=(908))).shape)
    
