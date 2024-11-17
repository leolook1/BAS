#coding=utf-8
import os
import cv2
from tqdm import tqdm
import metric as M
from sklearn.metrics import roc_auc_score

class AUC(object):
    def __init__(self):
        self.preds = []
        self.gts = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        # Flatten pred and gt arrays to 1D arrays to match the input format for AUC calculation
        pred, gt = _prepare_data(pred, gt)
        self.preds.extend(pred.flatten())
        self.gts.extend(gt.flatten())

    def get_results(self) -> dict:
        # Compute AUC
        auc_score = roc_auc_score(self.gts, self.preds)
        return dict(auc=auc_score)



FM = M.F1measure()
MAE = M.MAE()

mask_root = './GT/'
pred_root = './Result/'
mask_name_list = sorted(os.listdir(mask_root))
for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    FM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)
    auc_calculator.step(pred, mask)
fm = FM.get_results()['fm']
mae = MAE.get_results()['mae']
auc_calculator = AUC()

AUC = auc_calculator.get_results()

print(
    'MAE:', mae.round(3), '; ',
    'meanFm:', fm['curve'].mean().round(3), '; ',
    'maxFm:', fm['curve'].max().round(3),
    "AUC:", AUC['auc'])
)

with open("../result.txt", "a+") as f:
    print(
          'FM:', fm.round(3), '; ',
          'MAE:', mae.round(3), '; ',
          'AUC:', AUC.round(3), '; ',
          file=f
          )
