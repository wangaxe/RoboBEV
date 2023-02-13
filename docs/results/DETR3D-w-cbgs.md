<img src="../figs/logo.png" align="right" width="10%">

# RoboDet Benchmark

The official [nuScenes metric](https://www.nuscenes.org/object-detection/?externalData=all&mapData=all&modalities=Any) are considered in our benchmark:

- **Average Translation Error (ATE)** is the Euclidean center distance in 2D (units in meters). 
- **Average Scale Error (ASE)** is the 3D intersection-over-union (IoU) after aligning orientation and translation (1 − IoU).
- **Average Orientation Error (AOE)** is the smallest yaw angle difference between prediction and ground truth (radians). All angles are measured on a full 360-degree period except for barriers where they are measured on a 180-degree period.
- **Average Velocity Error (AVE)** is the absolute velocity error as the L2 norm of the velocity differences in 2D (m/s).
- **Average Attribute Error (AAE)** is defined as 1 minus attribute classification accuracy (1 − acc).
- **nuScenes Detection Score (NDS)**: $$\text{NDS} = \frac{1}{10} [5\text{mAP}+\sum_{\text{mTP}\in\mathbb{TP}} (1-\min(1, \text{mTP}))]$$

## DETR3D w cbgs

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Clean** | 0.4341 | 0.3494 | 0.7163 | 0.2682 | 0.3798 | 0.8421 | 0.1997 |
| **Motion Blur** | 0.2542    | 0.1385    | 0.8909     | 0.3355     | 0.6707     | 1.0682     | 0.2928     |
| **Color Quant** | 0.3235    | 0.2152    | 0.8571     | 0.2875     | 0.5350     | 0.9354     | 0.2400     |
| **Frame Lost** | 0.2685    | 0.0923    | 0.8268     | 0.3135     | 0.5042     | 0.8867     | 0.2455     |
| **Camera Crash** | 0.2991    | 0.1174    | 0.7932     | 0.2853     | 0.4575     | 0.8471     | 0.2131     |
| **Brightness** | 0.4154    | 0.3200    | 0.7357     | 0.2720     | 0.4086     | 0.8302     | 0.1990     |
| **Low Light** | 0.3233    | 0.2041    | 0.8105     | 0.2851     | 0.4918     | 0.9913     | 0.2347     |
| **Fog** | 0.4020    | 0.3012    | 0.7552     | 0.2710     | 0.4237     | 0.8302     | 0.2054     |
| **Snow** | 0.1925    | 0.0702    | 0.9246     | 0.3793     | 0.7648     | 1.2585     | 0.3577     |

## Experiment Log

Time: Sun Feb 12 09:52:47 2023

### Evaluating Snow

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2527    | 0.1262    | 0.8959     | 0.3077     | 0.6333     | 1.1899     | 0.2672     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1718    | 0.0503    | 0.9228     | 0.4092     | 0.8054     | 1.2890     | 0.3964     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1530    | 0.0342    | 0.9550     | 0.4210     | 0.8556     | 1.2965     | 0.4094     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1925    | 0.0702    | 0.9246     | 0.3793     | 0.7648     | 1.2585     | 0.3577     |

### Evaluating ColorQuant

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.4133    | 0.3226    | 0.7530     | 0.2692     | 0.3985     | 0.8464     | 0.2124     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3396    | 0.2307    | 0.8498     | 0.2804     | 0.4733     | 0.9189     | 0.2355     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2175    | 0.0925    | 0.9685     | 0.3130     | 0.7333     | 1.0408     | 0.2722     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3235    | 0.2152    | 0.8571     | 0.2875     | 0.5350     | 0.9354     | 0.2400     |

### Evaluating LowLight

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3406    | 0.2288    | 0.8007     | 0.2790     | 0.4849     | 0.9505     | 0.2226     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3417    | 0.2267    | 0.7994     | 0.2795     | 0.4734     | 0.9445     | 0.2199     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2878    | 0.1569    | 0.8315     | 0.2967     | 0.5170     | 1.0788     | 0.2616     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3233    | 0.2041    | 0.8105     | 0.2851     | 0.4918     | 0.9913     | 0.2347     |

Time: Fri Jan 20 20:59:44 2023

### Evaluating MotionBlur

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3758    | 0.2749    | 0.7891     | 0.2762     | 0.4583     | 0.8815     | 0.2115     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2218    | 0.0896    | 0.9283     | 0.3197     | 0.7191     | 1.1042     | 0.2629     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1650    | 0.0508    | 0.9554     | 0.4105     | 0.8346     | 1.2189     | 0.4040     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2542    | 0.1385    | 0.8909     | 0.3355     | 0.6707     | 1.0682     | 0.2928     |

### Evaluating Fog

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.4181    | 0.3199    | 0.7345     | 0.2692     | 0.3974     | 0.8136     | 0.2033     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.4053    | 0.3045    | 0.7508     | 0.2704     | 0.4160     | 0.8281     | 0.2038     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3826    | 0.2791    | 0.7802     | 0.2733     | 0.4578     | 0.8490     | 0.2092     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.4020    | 0.3012    | 0.7552     | 0.2710     | 0.4237     | 0.8302     | 0.2054     |

### Evaluating Brightness

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.4341    | 0.3440    | 0.7121     | 0.2698     | 0.3884     | 0.8130     | 0.1959     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.4181    | 0.3232    | 0.7334     | 0.2716     | 0.4057     | 0.8255     | 0.1984     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3940    | 0.2926    | 0.7617     | 0.2746     | 0.4318     | 0.8522     | 0.2026     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.4154    | 0.3200    | 0.7357     | 0.2720     | 0.4086     | 0.8302     | 0.1990     |

### Evaluating CameraCrash

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3399    | 0.1790    | 0.7659     | 0.2797     | 0.4204     | 0.8238     | 0.2064     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2733    | 0.0812    | 0.8138     | 0.2851     | 0.4587     | 0.8909     | 0.2243     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2840    | 0.0920    | 0.7998     | 0.2910     | 0.4935     | 0.8266     | 0.2085     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2991    | 0.1174    | 0.7932     | 0.2853     | 0.4575     | 0.8471     | 0.2131     |

### Evaluating FrameLost

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3554    | 0.2044    | 0.7517     | 0.2749     | 0.4039     | 0.8415     | 0.1965     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2606    | 0.0626    | 0.8209     | 0.2942     | 0.4963     | 0.8785     | 0.2171     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1894    | 0.0098    | 0.9077     | 0.3713     | 0.6125     | 0.9401     | 0.3229     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2685    | 0.0923    | 0.8268     | 0.3135     | 0.5042     | 0.8867     | 0.2455     |