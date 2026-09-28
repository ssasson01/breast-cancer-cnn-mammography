# breast-cancer-cnn-mammography
CNN-based breast cancer detection from mammography (CBIS-DDSM), comparing a baseline CNN against EfficientNetB0 transfer learning, evaluated on AUC-ROC and sensitivity.

**Demo:** https://breast-cancer-cnn-mammography-lyxpxpa4wdmtbtctmyyhfk.streamlit.app/

**Research prototype only - not a diagnostic tool.** Sensitivity on held-out test data is well below clinical benchmarks; do not use this for real clinical decisions.

## Results
Held-out CBIS-DDSM test set (n = 378):

| Model | AUC-ROC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Baseline CNN (from scratch, 64×64) | 0.6350 | 0.483 | 0.732 | 63.5% |
| **EfficientNetB0 (transfer learning, 224×224)** | **0.7092** | **0.544** | **0.732** | **65.9%** |

EfficientNetB0 uses a decision threshold of 0.591, chosen on the validation set with Youden's J statistic.

## Dataset
[CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) (Lee et al., 2017)

## Repository Structure
```
   ├── app.py                                                                # Streamlit app
   ├── requirements.txt
   ├── model/
   │   └── best_model.keras                                                  # Trained model
   ├── notebook/
   │   └── Deep_Learning_Breast_Cancer_Colab.ipynb                           # ipynb file from Google Colab
   └── samples/                                                               # 10 CBIS-DDSM test images to try in the app
```

## Running the app locally

```bash
git clone https://github.com/ssasson01/breast-cancer-cnn-mammography.git
cd breast-cancer-cnn-mammography
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and upload a mammogram mass image (JPEG or PNG).

## Reproducing the training
Open the notebook in Google Colab with a T4 GPU runtime. Add your Kaggle credentials as Colab secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`), then run the cells in order. Results can vary between runs because of non-deterministic GPU training and the small dataset.

## Limitations
- The model misses about 46% of malignant cases (sensitivity 0.544).
- Whole mammograms are downsampled to 224×224, so small masses can occupy only a few pixels.
- Small training set (1,054 images); results come from a single training run.

## License
- **Code:** MIT (see `LICENSE`)
- **Sample images in `samples/`:** CC BY-SA 3.0, from CBIS-DDSM (see `samples/README.md`)

## References
Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., & Rubin, D. L. (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177. https://doi.org/10.1038/sdata.2017.177

## Author

Shai Sasson. Deep Learning coursework project.
