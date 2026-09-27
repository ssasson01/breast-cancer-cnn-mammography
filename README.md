# breast-cancer-cnn-mammography
CNN-based breast cancer detection from mammography (CBIS-DDSM), comparing a baseline CNN against EfficientNetB0 transfer learning, evaluated on AUC-ROC and sensitivity.

**Demo:** https://breast-cancer-cnn-mammography-lyxpxpa4wdmtbtctmyyhfk.streamlit.app/

---

## Dataset
[CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) (Lee et al., 2017)

---

## Repository Structure

## Running the app locally

```bash
git clone https://github.com/ssasson01/breast-cancer-cnn-mammography.git
cd breast-cancer-cnn-mammography
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and upload a mammogram mass image (JPEG or PNG).

---

## Author

Shai Sasson. Deep Learning coursework project.
