# Breast Cancer Prediction from Mammograms

This project combines mammogram image analysis and lifestyle-related risk factors to predict the likelihood of breast cancer. It uses a multimodal neural network that considers both visual and clinical data to provide a more personalized prediction.

A Tkinter-based interface allows users to:
- Upload their own DICOM-format mammograms
- Enter lifestyle and family history information
- View prediction results
- Download a personalized PDF risk report with feature importance

---

## How it Works

1. **Mammogram Image Input**  
   DICOM images are preprocessed and passed through a CNN to extract visual patterns.

2. **Lifestyle & Risk Factor Input**  
   The user provides additional data such as:
   - Age
   - Smoking/alcohol history
   - Marital status
   - Family history of breast cancer
   - Menopause status
   - BMI range

3. **Multimodal Prediction**  
   The model merges both image and clinical inputs to predict whether the case is likely benign or malignant.

4. **SHAP Explainability**  
   SHAP values are used to show which lifestyle features most influenced the prediction.

5. **PDF Report**  
   A downloadable report is generated with:
   - Patient details
   - Prediction result
   - Risk factor analysis
   - Estimated future risk across different ages

---

## Project Structure

```
breast-cancer-prediction-from-mammograms/
│
├── gui/
│   └── tkinter_app.py               # Main Tkinter GUI
│
├── models/
│   └── model.h5                     # Trained multimodal model
│
├── dicom/
│   └── (sample DICOM files)         # Use your own .dcm files here
│
├── data/
│   ├── calc_case_description_test_set.csv
│   └── risk_factors/
│       ├── bcsc_part1.csv
│       ├── bcsc_part2.csv
│       └── bcsc_part3.csv
│
├── multimodal_mammogram_prediction.ipynb  # Full working notebook
├── multimodal_mammogram_prediction.py     # Python version of the notebook
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/breast-cancer-prediction-from-mammograms.git
cd breast-cancer-prediction-from-mammograms
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
python gui/tkinter_app.py
```

This will open the desktop GUI where you can:
- Upload mammogram DICOM files
- Fill out risk factors
- Generate a prediction and download a report

---

## Disclaimer

This is an experimental AI tool meant for academic and research purposes. It does not replace professional medical advice, diagnosis, or treatment.

---

## Questions?

Feel free to open an issue or drop a message if you'd like to collaborate or improve the app further.
