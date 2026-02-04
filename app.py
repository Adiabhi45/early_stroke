# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import plotly.express as px
# # import plotly.graph_objects as go
# # from plotly.subplots import make_subplots
# # import shap
# # import pickle
# # import joblib
# # import torch
# # import torch.nn as nn
# # from torchvision import transforms
# # from torchvision.models.segmentation import deeplabv3_resnet50
# # from tensorflow.keras.models import load_model
# # from PIL import Image
# # import cv2
# # from sklearn.preprocessing import StandardScaler, LabelEncoder
# # import io
# # import base64
# # import warnings
# # import os

# # # Suppress warnings
# # warnings.filterwarnings('ignore')
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # # Page configuration
# # st.set_page_config(
# #     page_title="🧠 AI Stroke Detection System",
# #     page_icon="🧠",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # Custom CSS for MERN-like styling
# # st.markdown("""
# # <style>
# #     /* Main background and text */
# #     .main {
# #         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# #         color: #ffffff;
# #     }
    
# #     .stApp {
# #         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# #     }
    
# #     /* Custom containers */
# #     .main-container {
# #         background: rgba(255, 255, 255, 0.1);
# #         backdrop-filter: blur(10px);
# #         border-radius: 15px;
# #         padding: 30px;
# #         margin: 20px 0;
# #         border: 1px solid rgba(255, 255, 255, 0.2);
# #         box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
# #     }
    
# #     .result-container {
# #         background: linear-gradient(145deg, #ff6b6b, #ee5a24);
# #         border-radius: 15px;
# #         padding: 25px;
# #         margin: 15px 0;
# #         box-shadow: 0 10px 25px rgba(0,0,0,0.2);
# #         color: white;
# #         text-align: center;
# #     }
    
# #     .normal-container {
# #         background: linear-gradient(145deg, #4ecdc4, #44a08d);
# #         border-radius: 15px;
# #         padding: 25px;
# #         margin: 15px 0;
# #         box-shadow: 0 10px 25px rgba(0,0,0,0.2);
# #         color: white;
# #         text-align: center;
# #     }
    
# #     .warning-container {
# #         background: linear-gradient(145deg, #feca57, #ff9ff3);
# #         border-radius: 15px;
# #         padding: 25px;
# #         margin: 15px 0;
# #         box-shadow: 0 10px 25px rgba(0,0,0,0.2);
# #         color: white;
# #         text-align: center;
# #     }
    
# #     /* Headers */
# #     h1 {
# #         color: #ffffff;
# #         text-align: center;
# #         font-size: 3rem;
# #         margin-bottom: 2rem;
# #         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
# #     }
    
# #     h2, h3 {
# #         color: #ffffff;
# #         text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
# #     }
    
# #     /* Buttons */
# #     .stButton > button {
# #         background: linear-gradient(145deg, #667eea, #764ba2);
# #         color: white;
# #         border: none;
# #         border-radius: 25px;
# #         padding: 15px 30px;
# #         font-size: 16px;
# #         font-weight: bold;
# #         box-shadow: 0 5px 15px rgba(0,0,0,0.2);
# #         transition: all 0.3s ease;
# #     }
    
# #     .stButton > button:hover {
# #         transform: translateY(-2px);
# #         box-shadow: 0 8px 25px rgba(0,0,0,0.3);
# #     }
    
# #     /* File uploader */
# #     .stFileUploader > div > div {
# #         background: rgba(255, 255, 255, 0.1);
# #         border-radius: 15px;
# #         border: 2px dashed rgba(255, 255, 255, 0.3);
# #         padding: 20px;
# #     }
    
# #     /* Sidebar */
# #     .css-1d391kg {
# #         background: rgba(255, 255, 255, 0.1);
# #         backdrop-filter: blur(10px);
# #     }
    
# #     /* Metrics */
# #     .metric-container {
# #         background: rgba(255, 255, 255, 0.1);
# #         backdrop-filter: blur(10px);
# #         border-radius: 10px;
# #         padding: 20px;
# #         text-align: center;
# #         margin: 10px 0;
# #     }
    
# #     /* Input fields */
# #     .stNumberInput > div > div > input {
# #         background: rgba(255, 255, 255, 0.1);
# #         color: white;
# #         border: 1px solid rgba(255, 255, 255, 0.3);
# #         border-radius: 8px;
# #     }
    
# #     .stSelectbox > div > div > select {
# #         background: rgba(255, 255, 255, 0.1);
# #         color: white;
# #         border: 1px solid rgba(255, 255, 255, 0.3);
# #         border-radius: 8px;
# #     }
# # </style>
# # """, unsafe_allow_html=True)

# # # Load models and preprocessors
# # @st.cache_resource
# # def load_models():
# #     """Load all models with proper error handling"""
# #     models_dict = {}
    
# #     try:
# #         # Load Random Forest model (joblib format)
# #         st.write("Loading Random Forest model...")
# #         try:
# #             models_dict['rf_model'] = joblib.load('random_forest_model.joblib')
# #             st.success("✅ Random Forest model loaded")
# #         except:
# #             # Fallback to .pkl extension
# #             rf_model=joblib.load('random_forest_model.pkl')
# #             models_dict['rf_model'] = joblib.load('random_forest_model.pkl')
# #             st.success("✅ Random Forest model loaded (pkl)")
# #     except Exception as e:
# #         st.error(f"❌ Failed to load Random Forest model: {e}")
# #         return None
    
# #     try:
# #         # Load Scaler (joblib format)
# #         st.write("Loading Scaler...")
# #         try:
# #             scaler = joblib.load('scaler.joblib')
# #             models_dict['scaler'] = joblib.load('scaler.joblib')
# #             st.success("✅ Scaler loaded")
# #         except:
# #             # Fallback to .pkl extension  
# #             models_dict['scaler'] = joblib.load('scaler.pkl')
# #             st.success("✅ Scaler loaded (pkl)")
# #     except Exception as e:
# #         st.error(f"❌ Failed to load Scaler: {e}")
# #         return None
    
# #     try:
# #         # Load Label Encoders (pickle format)
# #         st.write("Loading Label Encoders...")
# #         with open('label_encoders.pkl', 'rb') as f:
# #             models_dict['label_encoders'] = pickle.load(f)
# #         st.success("✅ Label Encoders loaded")
# #     except Exception as e:
# #         st.error(f"❌ Failed to load Label Encoders: {e}")
# #         models_dict['label_encoders'] = None
    
# #     try:
# #         # Load SHAP Explainer (or create from CSV data)
# #         st.write("Loading SHAP Explainer...")
# #         try:
# #             with open('random_forest_shap_explainer.pkl', 'rb') as f:
# #                 models_dict['shap_explainer'] = pickle.load(f)
# #             st.success("✅ SHAP Explainer loaded")
# #         except:
# #             try:
# #                 # Fallback filename
# #                 with open('random_forest_shap_explainer.pkl', 'rb') as f:
# #                     models_dict['shap_explainer'] = pickle.load(f)
# #                 st.success("✅ SHAP Explainer loaded (fallback)")
# #             except Exception as fallback_error:
# #                 # Create new SHAP explainer from CSV data
# #                 st.warning(f"⚠️ Could not load saved SHAP explainer: {fallback_error}")
# #                 st.info("Loading CSV data to create new SHAP explainer...")
# #                 try:
# #                     # Try to load CSV data
# #                     csv_files = ['stroke_data.csv', 'data.csv', 'stroke_dataset.csv', 'clinical_data.csv']
# #                     data = None
                    
# #                     for csv_file in csv_files:
# #                         try:
# #                             data = pd.read_csv(csv_file)
# #                             st.info(f"✅ Loaded data from {csv_file}")
# #                             break
# #                         except:
# #                             continue
                    
# #                     if data is not None:
# #                         # Prepare data same way as training
# #                         # Apply categorical encoding
# #                         cat_cols = ['sex', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
                        
# #                         # Create sample for SHAP (use subset for performance)
# #                         sample_size = min(1000, len(data))
# #                         sample_data = data.sample(n=sample_size, random_state=42).copy()
                        
# #                         # Manual encoding to match your training
# #                         # Assuming your CSV has numerical encoded values already
# #                         X_sample = sample_data.drop('stroke', axis=1)
                        
# #                         # Scale the sample data using the loaded scaler
# #                         num_cols = ['age', 'avg_glucose_level', 'bmi']
# #                         X_sample_scaled = X_sample.copy()
# #                         X_sample_scaled[num_cols] = scaler.transform(X_sample[num_cols])
                        
# #                         # Create SHAP explainer with sample data
# #                         models_dict['shap_explainer'] = shap.TreeExplainer(rf_model, X_sample_scaled)
# #                         models_dict['shap_background_data'] = X_sample_scaled
# #                         st.success("✅ SHAP Explainer created from CSV data")
# #                     else:
# #                         st.warning("⚠️ No CSV data found. SHAP analysis will use feature importance.")
# #                         models_dict['shap_explainer'] = None
# #                         models_dict['shap_background_data'] = None
                        
# #                 except Exception as csv_error:
# #                     st.warning(f"Could not create SHAP from CSV: {csv_error}")
# #                     models_dict['shap_explainer'] = None
# #                     models_dict['shap_background_data'] = None
                    
# #     except Exception as e:
# #         st.error(f"❌ Failed to load SHAP Explainer: {e}")
# #         st.info("Will use feature importance for explanations")
# #         models_dict['shap_explainer'] = None
# #         models_dict['shap_background_data'] = None
    
# #     try:
# #         # Load Image Classification Model (Keras format)
# #         st.write("Loading Image Classification Model...")
# #         models_dict['image_model'] = load_model('best_model.keras')
# #         st.success("✅ Image Classification model loaded")
# #     except Exception as e:
# #         st.error(f"❌ Failed to load Image Classification model: {e}")
# #         models_dict['image_model'] = None
    
# #     try:
# #         # Load Segmentation Model (PyTorch format)
# #         st.write("Loading Segmentation Model...")
# #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# #         # Load the state dict first to check its structure
# #         checkpoint = torch.load('best_deeplabv3.pth', map_location=device)
        
# #         # Check if aux_classifier keys exist
# #         has_aux_classifier = any(key.startswith('aux_classifier') for key in checkpoint.keys())
        
# #         if has_aux_classifier:
# #             # Create model WITH aux_classifier since the saved model has it
# #             seg_model = deeplabv3_resnet50(pretrained=False, aux_loss=True)
# #         else:
# #             # Create model WITHOUT aux_classifier
# #             seg_model = deeplabv3_resnet50(pretrained=False, aux_loss=False)
        
# #         # Modify the final layer for binary segmentation
# #         seg_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        
# #         # If model was trained with aux_classifier but we want to use without it,
# #         # we can filter out those keys
# #         if has_aux_classifier:
# #             # Option 1: Load with aux_classifier
# #             try:
# #                 seg_model.load_state_dict(checkpoint, strict=True)
# #                 st.success("✅ Segmentation model loaded with aux_classifier")
# #             except:
# #                 # Option 2: Filter out aux_classifier and load without it
# #                 seg_model = deeplabv3_resnet50(pretrained=False, aux_loss=False)
# #                 seg_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
                
# #                 filtered_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('aux_classifier')}
# #                 seg_model.load_state_dict(filtered_state_dict, strict=False)
# #                 st.success("✅ Segmentation model loaded without aux_classifier")
# #         else:
# #             seg_model.load_state_dict(checkpoint, strict=True)
# #             st.success("✅ Segmentation model loaded")
        
# #         seg_model.to(device)
# #         seg_model.eval()
# #         models_dict['seg_model'] = seg_model
# #         models_dict['device'] = device
        
# #     except Exception as e:
# #         st.error(f"❌ Failed to load Segmentation model: {e}")
# #         st.info("Segmentation functionality will be disabled")
# #         models_dict['seg_model'] = None
# #         models_dict['device'] = torch.device("cpu")
    
# #     return models_dict

# # # Image preprocessing functions
# # def preprocess_image_classification(image):
# #     """Preprocess image for classification model"""
# #     try:
# #         image = image.convert('L')  # Convert to grayscale
# #         image = image.resize((224, 224))
# #         image_array = np.array(image) / 255.0
# #         image_array = np.expand_dims(image_array, axis=0)
# #         image_array = np.expand_dims(image_array, axis=-1)
# #         return image_array
# #     except Exception as e:
# #         st.error(f"Error preprocessing image for classification: {e}")
# #         return None

# # def preprocess_image_segmentation(image, device):
# #     """Preprocess image for segmentation model"""
# #     try:
# #         transform = transforms.Compose([
# #             transforms.Resize((224, 224)),
# #             transforms.ToTensor(),
# #             transforms.Normalize(mean=[0.485], std=[0.229])
# #         ])
        
# #         if image.mode != 'L':
# #             image = image.convert('L')
        
# #         image_tensor = transform(image).unsqueeze(0).to(device)
# #         return image_tensor
# #     except Exception as e:
# #         st.error(f"Error preprocessing image for segmentation: {e}")
# #         return None

# # def predict_clinical_data(data, rf_model, scaler, label_encoders):
# #     """Predict stroke from clinical data"""
# #     try:
# #         # Create input dataframe
# #         input_data = pd.DataFrame([data])
        
# #         # Manual encoding based on your data structure
# #         sex_mapping = {'Male': 1, 'Female': 0, 'Other': 2}
# #         input_data['sex'] = sex_mapping.get(data['sex'], 0)
        
# #         married_mapping = {'Yes': 1, 'No': 0}
# #         input_data['ever_married'] = married_mapping.get(data['ever_married'], 0)
        
# #         work_mapping = {
# #             'Never_worked': 0, 
# #             'children': 1, 
# #             'Govt_job': 2, 
# #             'Private': 3, 
# #             'Self-employed': 4
# #         }
# #         input_data['work_type'] = work_mapping.get(data['work_type'], 3)
        
# #         residence_mapping = {'Rural': 0, 'Urban': 1}
# #         input_data['Residence_type'] = residence_mapping.get(data['Residence_type'], 1)
        
# #         smoking_mapping = {
# #             'never smoked': 0,
# #             'formerly smoked': 1, 
# #             'smokes': 1
# #         }
# #         input_data['smoking_status'] = smoking_mapping.get(data['smoking_status'], 0)
        
# #         # Scale numerical features
# #         num_cols = ['age', 'avg_glucose_level', 'bmi']
# #         input_data[num_cols] = scaler.transform(input_data[num_cols])
        
# #         # Predict
# #         prediction = rf_model.predict(input_data)[0]
# #         probability = rf_model.predict_proba(input_data)[0]
        
# #         return prediction, probability
# #     except Exception as e:
# #         st.error(f"Error in clinical prediction: {str(e)}")
# #         return None, None

# # def predict_image_classification(image, model):
# #     """Predict stroke type from CT scan"""
# #     try:
# #         if model is None:
# #             st.warning("Image classification model not available")
# #             return "Model Unavailable", 0.0, [0.33, 0.33, 0.34]
        
# #         processed_image = preprocess_image_classification(image)
# #         if processed_image is None:
# #             return None, None, None
            
# #         prediction = model.predict(processed_image, verbose=0)
# #         predicted_class = np.argmax(prediction[0])
# #         confidence = np.max(prediction[0])
        
# #         # Map classes (adjust based on your model)
# #         class_names = ['Hemorrhage', 'No Stroke', 'Ischemic']
        
# #         return class_names[predicted_class], confidence, prediction[0]
# #     except Exception as e:
# #         st.error(f"Error in image classification: {str(e)}")
# #         return None, None, None

# # def segment_stroke_region(image, model, device):
# #     """Generate segmentation mask for stroke region"""
# #     try:
# #         if model is None:
# #             st.warning("Segmentation model not available")
# #             return None
            
# #         processed_image = preprocess_image_segmentation(image, device)
# #         if processed_image is None:
# #             return None
        
# #         with torch.no_grad():
# #             output = model(processed_image)
# #             mask = torch.sigmoid(output['out']).cpu().numpy()[0, 0]
        
# #         return mask
# #     except Exception as e:
# #         st.error(f"Error in segmentation: {str(e)}")
# #         return None

# # def main():
# #     # App header
# #     st.markdown("<h1>🧠 AI-Powered Stroke Detection System</h1>", unsafe_allow_html=True)
# #     st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #ffffff; margin-bottom: 2rem;'>Advanced Medical AI for Early Stroke Detection using Clinical Data and CT Scans</p>", unsafe_allow_html=True)
    
# #     # Load models
# #     with st.spinner("Loading AI models..."):
# #         models_dict = load_models()
    
# #     if models_dict is None:
# #         st.error("Critical error: Could not load essential models.")
# #         st.stop()
    
# #     # Extract models from dictionary
# #     rf_model = models_dict.get('rf_model')
# #     scaler = models_dict.get('scaler')
# #     label_encoders = models_dict.get('label_encoders')
# #     shap_explainer = models_dict.get('shap_explainer')
# #     shap_background_data = models_dict.get('shap_background_data')
# #     image_model = models_dict.get('image_model')
# #     seg_model = models_dict.get('seg_model')
# #     device = models_dict.get('device', torch.device("cpu"))
    
# #     # Show model status
# #     with st.expander("📊 Model Status", expanded=False):
# #         col1, col2, col3 = st.columns(3)
# #         with col1:
# #             st.write("**Clinical Models:**")
# #             st.write(f"Random Forest: {'✅' if rf_model else '❌'}")
# #             st.write(f"Scaler: {'✅' if scaler else '❌'}")
# #             st.write(f"Label Encoders: {'✅' if label_encoders else '❌'}")
# #         with col2:
# #             st.write("**Image Models:**")
# #             st.write(f"Classification: {'✅' if image_model else '❌'}")
# #             st.write(f"Segmentation: {'✅' if seg_model else '❌'}")
# #         with col3:
# #             st.write("**Explainability:**")
# #             st.write(f"SHAP: {'✅' if shap_explainer else '❌'}")
# #             st.write(f"Background Data: {'✅' if shap_background_data is not None else '❌'}")
# #             st.write(f"Device: {device}")
    
# #     # Sidebar
# #     st.sidebar.markdown("## 📋 Patient Information")
# #     st.sidebar.markdown("---")
    
# #     # Clinical data input
# #     col1, col2 = st.columns(2)
    
# #     with col1:
# #         st.markdown('<div class="main-container">', unsafe_allow_html=True)
# #         st.markdown("### 👤 Clinical Data Input")
        
# #         age = st.number_input("Age", min_value=0, max_value=120, value=50)
# #         sex = st.selectbox("Sex", ["Male", "Female"])
# #         hypertension = st.selectbox("Hypertension", ["No", "Yes"])
# #         heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
# #         ever_married = st.selectbox("Ever Married", ["No", "Yes"])
# #         work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
# #         residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
# #         avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=100.0)
# #         bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
# #         smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])
        
# #         st.markdown('</div>', unsafe_allow_html=True)
    
# #     with col2:
# #         st.markdown('<div class="main-container">', unsafe_allow_html=True)
# #         st.markdown("### 🔬 CT Scan Upload")
        
# #         uploaded_file = st.file_uploader(
# #             "Upload CT Scan Image",
# #             type=['png', 'jpg', 'jpeg'],
# #             help="Upload a CT scan image for analysis"
# #         )
        
# #         if uploaded_file is not None:
# #             image = Image.open(uploaded_file)
# #             st.image(image, caption="Uploaded CT Scan", use_column_width=True)
        
# #         st.markdown('</div>', unsafe_allow_html=True)
    
# #     # Analysis button
# #     if st.button("🔍 Analyze Patient Data", key="analyze_btn"):
# #         # Prepare clinical data
# #         clinical_data = {
# #             'age': age,
# #             'sex': sex,
# #             'hypertension': 1 if hypertension == "Yes" else 0,
# #             'heart_disease': 1 if heart_disease == "Yes" else 0,
# #             'ever_married': ever_married,
# #             'work_type': work_type,
# #             'Residence_type': residence_type,
# #             'avg_glucose_level': avg_glucose_level,
# #             'bmi': bmi,
# #             'smoking_status': smoking_status
# #         }
        
# #         # Clinical prediction
# #         if rf_model and scaler:
# #             clinical_pred, clinical_prob = predict_clinical_data(clinical_data, rf_model, scaler, label_encoders)
# #         else:
# #             st.error("Clinical analysis not available - missing models")
# #             clinical_pred, clinical_prob = None, None
        
# #         # Image prediction
# #         image_pred, image_conf, image_probs = None, None, None
# #         if uploaded_file is not None and image_model:
# #             image_pred, image_conf, image_probs = predict_image_classification(image, image_model)
# #         elif uploaded_file is None:
# #             st.warning("Please upload a CT scan image for complete analysis")
        
# #         # Results display
# #         if clinical_pred is not None:
# #             st.markdown("## 📊 Analysis Results")
            
# #             col1, col2, col3 = st.columns(3)
            
# #             with col1:
# #                 st.markdown('<div class="metric-container">', unsafe_allow_html=True)
# #                 st.metric("Clinical Analysis", 
# #                          "Stroke Risk" if clinical_pred == 1 else "Normal", 
# #                          f"{clinical_prob[1]*100:.1f}% confidence")
# #                 st.markdown('</div>', unsafe_allow_html=True)
            
# #             with col2:
# #                 if image_pred:
# #                     st.markdown('<div class="metric-container">', unsafe_allow_html=True)
# #                     st.metric("CT Scan Analysis", 
# #                              image_pred, 
# #                              f"{image_conf*100:.1f}% confidence")
# #                     st.markdown('</div>', unsafe_allow_html=True)
# #                 else:
# #                     st.markdown('<div class="metric-container">', unsafe_allow_html=True)
# #                     st.metric("CT Scan Analysis", "Not Available", "No image uploaded")
# #                     st.markdown('</div>', unsafe_allow_html=True)
            
# #             with col3:
# #                 # Combined analysis
# #                 if clinical_pred == 0 and (image_pred == "No Stroke" or image_pred is None):
# #                     final_result = "Normal"
# #                 elif clinical_pred == 1 and (image_pred == "No Stroke" or image_pred is None):
# #                     final_result = "Clinical Risk - Monitor"
# #                 elif image_pred and image_pred != "No Stroke":
# #                     final_result = f"Stroke Detected - {image_pred}"
# #                 else:
# #                     final_result = "Clinical Analysis Only"
                
# #                 st.markdown('<div class="metric-container">', unsafe_allow_html=True)
# #                 st.metric("Final Diagnosis", final_result, "Combined Analysis")
# #                 st.markdown('</div>', unsafe_allow_html=True)
            
# #             # SHAP Analysis
# #             if shap_explainer or rf_model:
# #                 st.markdown('<div class="main-container">', unsafe_allow_html=True)
# #                 st.markdown("### 🎯 Clinical Data Explanation (SHAP)")
                
# #                 try:
# #                     # Create SHAP explainer if not loaded (using fresh approach)
# #                     if shap_explainer is None:
# #                         st.info("Creating new SHAP explainer from Random Forest model...")
                        
# #                         # Create a simple explainer with just the model
# #                         shap_explainer = shap.TreeExplainer(rf_model)
                    
# #                     # Prepare data for SHAP using same encoding as prediction
# #                     input_df = pd.DataFrame([clinical_data])
                    
# #                     # Apply same manual encoding
# #                     sex_mapping = {'Male': 1, 'Female': 0, 'Other': 2}
# #                     input_df['sex'] = sex_mapping.get(clinical_data['sex'], 0)
                    
# #                     married_mapping = {'Yes': 1, 'No': 0}
# #                     input_df['ever_married'] = married_mapping.get(clinical_data['ever_married'], 0)
                    
# #                     work_mapping = {
# #                         'Never_worked': 0, 
# #                         'children': 1, 
# #                         'Govt_job': 2, 
# #                         'Private': 3, 
# #                         'Self-employed': 4
# #                     }
# #                     input_df['work_type'] = work_mapping.get(clinical_data['work_type'], 3)
                    
# #                     residence_mapping = {'Rural': 0, 'Urban': 1}
# #                     input_df['Residence_type'] = residence_mapping.get(clinical_data['Residence_type'], 1)
                    
# #                     smoking_mapping = {
# #                         'never smoked': 0,
# #                         'formerly smoked': 1, 
# #                         'smokes': 1
# #                     }
# #                     input_df['smoking_status'] = smoking_mapping.get(clinical_data['smoking_status'], 0)
                    
# #                     num_cols = ['age', 'avg_glucose_level', 'bmi']
# #                     input_df[num_cols] = scaler.transform(input_df[num_cols])
                    
# #                     # Generate SHAP values for this single prediction
# #                     shap_values = shap_explainer.shap_values(input_df)
                    
# #                     # Use feature names for better readability
# #                     feature_names = ['Age', 'Sex', 'Hypertension', 'Heart Disease', 'Ever Married', 
# #                                    'Work Type', 'Residence Type', 'Glucose Level', 'BMI', 'Smoking Status']
                    
# #                     # Create a more robust SHAP visualization
# #                     if isinstance(shap_values, list) and len(shap_values) > 1:
# #                         # Binary classification - use stroke class (index 1)
# #                         values_to_plot = shap_values[1][0]  # First (and only) sample, stroke class
# #                         expected_value = shap_explainer.expected_value[1]
# #                     else:
# #                         # Single output
# #                         values_to_plot = shap_values[0]  # First sample
# #                         expected_value = shap_explainer.expected_value
                    
# #                     # Create bar plot manually for better control
# #                     fig, ax = plt.subplots(figsize=(10, 6))
                    
# #                     # Sort features by absolute SHAP value
# #                     abs_values = np.abs(values_to_plot)
# #                     sorted_indices = np.argsort(abs_values)
                    
# #                     # Plot horizontal bar chart
# #                     colors = ['red' if v < 0 else 'blue' for v in values_to_plot[sorted_indices]]
# #                     bars = ax.barh(range(len(sorted_indices)), values_to_plot[sorted_indices], color=colors, alpha=0.7)
                    
# #                     # Customize plot
# #                     ax.set_yticks(range(len(sorted_indices)))
# #                     ax.set_yticklabels([feature_names[i] for i in sorted_indices])
# #                     ax.set_xlabel('SHAP Value (Impact on Model Output)')
# #                     ax.set_title(f'Feature Impact on Stroke Prediction\n(Expected Value: {expected_value:.3f})')
# #                     ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.8)
                    
# #                     # Add value labels on bars
# #                     for i, bar in enumerate(bars):
# #                         width = bar.get_width()
# #                         ax.text(width + (0.001 if width >= 0 else -0.001), bar.get_y() + bar.get_height()/2, 
# #                                f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
                    
# #                     plt.tight_layout()
# #                     st.pyplot(fig)
                    
# #                     # Add interpretation
# #                     st.markdown("**🔍 Interpretation:**")
# #                     st.markdown("- **Blue bars**: Features increasing stroke risk")
# #                     st.markdown("- **Red bars**: Features decreasing stroke risk") 
# #                     st.markdown("- **Longer bars**: Stronger impact on prediction")
                    
# #                 except Exception as e:
# #                     st.warning(f"SHAP analysis error: {str(e)}")
                    
# #                     # Fallback: show feature importance
# #                     st.info("Showing feature importance instead:")
# #                     feature_names = ['Age', 'Sex', 'Hypertension', 'Heart Disease', 'Ever Married', 
# #                                    'Work Type', 'Residence Type', 'Glucose Level', 'BMI', 'Smoking Status']
# #                     importance = rf_model.feature_importances_
                    
# #                     fig, ax = plt.subplots(figsize=(10, 6))
# #                     indices = np.argsort(importance)[::-1]
# #                     bars = plt.bar(range(len(importance)), importance[indices], color='lightcoral', alpha=0.7)
# #                     plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
# #                     plt.title('Feature Importance from Random Forest Model')
# #                     plt.xlabel('Features')
# #                     plt.ylabel('Importance')
                    
# #                     # Add value labels on bars
# #                     for i, bar in enumerate(bars):
# #                         height = bar.get_height()
# #                         ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
# #                                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                    
# #                     plt.tight_layout()
# #                     st.pyplot(fig)
                
# #                 st.markdown('</div>', unsafe_allow_html=True)
            
# #             # Image Analysis Results
# #             if image_probs is not None:
# #                 st.markdown('<div class="main-container">', unsafe_allow_html=True)
# #                 st.markdown("### 🖼️ Image Analysis Results")
                
# #                 # Create probability chart
# #                 fig = go.Figure(data=[
# #                     go.Bar(x=['Hemorrhage', 'No Stroke', 'Ischemic'], 
# #                            y=image_probs*100,
# #                            marker_color=['#ff6b6b', '#4ecdc4', '#feca57'])
# #                 ])
# #                 fig.update_layout(
# #                     title="CT Scan Classification Probabilities",
# #                     yaxis_title="Confidence (%)",
# #                     template="plotly_dark",
# #                     paper_bgcolor='rgba(0,0,0,0)',
# #                     plot_bgcolor='rgba(0,0,0,0)'
# #                 )
# #                 st.plotly_chart(fig, use_container_width=True)
                
# #                 st.markdown('</div>', unsafe_allow_html=True)
            
# #             # Segmentation for stroke cases
# #             if image_pred and image_pred != "No Stroke" and seg_model and uploaded_file:
# #                 st.markdown('<div class="main-container">', unsafe_allow_html=True)
# #                 st.markdown("### 🎯 Stroke Region Segmentation")
                
# #                 mask = segment_stroke_region(image, seg_model, device)
                
# #                 if mask is not None:
# #                     col1, col2, col3 = st.columns(3)
                    
# #                     with col1:
# #                         st.markdown("**Original Image**")
# #                         st.image(image, use_column_width=True)
                    
# #                     with col2:
# #                         st.markdown("**Segmentation Mask**")
# #                         fig, ax = plt.subplots()
# #                         ax.imshow(mask, cmap='hot')
# #                         ax.axis('off')
# #                         st.pyplot(fig)
                    
# #                     with col3:
# #                         st.markdown("**Overlay**")
# #                         # Create overlay
# #                         img_array = np.array(image.convert('RGB'))
# #                         mask_colored = plt.cm.hot(mask)[:, :, :3]
# #                         overlay = 0.7 * img_array/255 + 0.3 * mask_colored
                        
# #                         fig, ax = plt.subplots()
# #                         ax.imshow(overlay)
# #                         ax.axis('off')
# #                         st.pyplot(fig)
                
# #                 st.markdown('</div>', unsafe_allow_html=True)
            
# #             # Final recommendation
# #             st.markdown('<div class="main-container">', unsafe_allow_html=True)
# #             st.markdown("### 💡 Medical Recommendation")
            
# #             if "Normal" in final_result:
# #                 st.success("✅ No immediate stroke risk detected. Continue regular health monitoring.")
# #             elif "Monitor" in final_result:
# #                 st.warning("⚠️ Clinical factors suggest increased stroke risk. Recommend consultation with healthcare provider.")
# #             elif "Stroke Detected" in final_result:
# #                 st.error("🚨 Potential stroke detected. Immediate medical attention required!")
# #             else:
# #                 st.info("ℹ️ Analysis completed with available models. Upload CT scan for complete assessment.")
            
# #             st.markdown('</div>', unsafe_allow_html=True)
    
# #     # Footer
# #     st.markdown("---")
# #     st.markdown(
# #         "<p style='text-align: center; color: #ffffff; opacity: 0.7;'>"
# #         "🏥 This AI system is for screening purposes only. Always consult healthcare professionals for medical decisions."
# #         "</p>", 
# #         unsafe_allow_html=True
# #     )

# # if __name__ == "__main__":
# #     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import shap
# import pickle
# import joblib
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torchvision.models.segmentation import deeplabv3_resnet50
# from tensorflow.keras.models import load_model
# from PIL import Image
# import cv2
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import io
# import base64
# import warnings
# import os

# # Suppress warnings
# warnings.filterwarnings('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # Page configuration
# st.set_page_config(
#     page_title="🧠 AI Stroke Detection System",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for MERN-like styling
# st.markdown("""
# <style>
#     /* Main background and text */
#     .main {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: #ffffff;
#     }
    
#     .stApp {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     }
    
#     /* Custom containers */
#     .main-container {
#         background: rgba(255, 255, 255, 0.1);
#         backdrop-filter: blur(10px);
#         border-radius: 15px;
#         padding: 30px;
#         margin: 20px 0;
#         border: 1px solid rgba(255, 255, 255, 0.2);
#         box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
#     }
    
#     .result-container {
#         background: linear-gradient(145deg, #ff6b6b, #ee5a24);
#         border-radius: 15px;
#         padding: 25px;
#         margin: 15px 0;
#         box-shadow: 0 10px 25px rgba(0,0,0,0.2);
#         color: white;
#         text-align: center;
#     }
    
#     .normal-container {
#         background: linear-gradient(145deg, #4ecdc4, #44a08d);
#         border-radius: 15px;
#         padding: 25px;
#         margin: 15px 0;
#         box-shadow: 0 10px 25px rgba(0,0,0,0.2);
#         color: white;
#         text-align: center;
#     }
    
#     .warning-container {
#         background: linear-gradient(145deg, #feca57, #ff9ff3);
#         border-radius: 15px;
#         padding: 25px;
#         margin: 15px 0;
#         box-shadow: 0 10px 25px rgba(0,0,0,0.2);
#         color: white;
#         text-align: center;
#     }
    
#     /* Headers */
#     h1 {
#         color: #ffffff;
#         text-align: center;
#         font-size: 3rem;
#         margin-bottom: 2rem;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#     }
    
#     h2, h3 {
#         color: #ffffff;
#         text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
#     }
    
#     /* Buttons */
#     .stButton > button {
#         background: linear-gradient(145deg, #667eea, #764ba2);
#         color: white;
#         border: none;
#         border-radius: 25px;
#         padding: 15px 30px;
#         font-size: 16px;
#         font-weight: bold;
#         box-shadow: 0 5px 15px rgba(0,0,0,0.2);
#         transition: all 0.3s ease;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 8px 25px rgba(0,0,0,0.3);
#     }
    
#     /* File uploader */
#     .stFileUploader > div > div {
#         background: rgba(255, 255, 255, 0.1);
#         border-radius: 15px;
#         border: 2px dashed rgba(255, 255, 255, 0.3);
#         padding: 20px;
#     }
    
#     /* Sidebar */
#     .css-1d391kg {
#         background: rgba(255, 255, 255, 0.1);
#         backdrop-filter: blur(10px);
#     }
    
#     /* Metrics */
#     .metric-container {
#         background: rgba(255, 255, 255, 0.1);
#         backdrop-filter: blur(10px);
#         border-radius: 10px;
#         padding: 20px;
#         text-align: center;
#         margin: 10px 0;
#     }
    
#     /* Input fields */
#     .stNumberInput > div > div > input {
#         background: rgba(255, 255, 255, 0.1);
#         color: white;
#         border: 1px solid rgba(255, 255, 255, 0.3);
#         border-radius: 8px;
#     }
    
#     .stSelectbox > div > div > select {
#         background: rgba(255, 255, 255, 0.1);
#         color: white;
#         border: 1px solid rgba(255, 255, 255, 0.3);
#         border-radius: 8px;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Load models and preprocessors
# @st.cache_resource
# def load_models():
#     """Load all models with proper error handling"""
#     models_dict = {}
    
#     try:
#         # Load Random Forest model (joblib format)
#         st.write("Loading Random Forest model...")
#         try:
#             models_dict['rf_model'] = joblib.load('random_forest_model.joblib')
#             st.success("✅ Random Forest model loaded")
#         except:
#             # Fallback to .pkl extension
#             models_dict['rf_model'] = joblib.load('random_forest_model.pkl')
#             st.success("✅ Random Forest model loaded (pkl)")
#     except Exception as e:
#         st.error(f"❌ Failed to load Random Forest model: {e}")
#         return None
    
#     try:
#         # Load Scaler (joblib format)
#         st.write("Loading Scaler...")
#         try:
#             models_dict['scaler'] = joblib.load('scaler.joblib')
#             st.success("✅ Scaler loaded")
#         except:
#             # Fallback to .pkl extension  
#             models_dict['scaler'] = joblib.load('scaler.pkl')
#             st.success("✅ Scaler loaded (pkl)")
#     except Exception as e:
#         st.error(f"❌ Failed to load Scaler: {e}")
#         return None
    
#     try:
#         # Load Label Encoders (pickle format)
#         st.write("Loading Label Encoders...")
#         with open('label_encoders.pkl', 'rb') as f:
#             models_dict['label_encoders'] = pickle.load(f)
#         st.success("✅ Label Encoders loaded")
#     except Exception as e:
#         st.error(f"❌ Failed to load Label Encoders: {e}")
#         models_dict['label_encoders'] = None
    
#     try:
#         # Load SHAP Explainer (or create from CSV data)
#         st.write("Loading SHAP Explainer...")
#         try:
#             with open('random_forest_shap_explainer.pkl', 'rb') as f:
#                 models_dict['shap_explainer'] = pickle.load(f)
#             st.success("✅ SHAP Explainer loaded")
#         except:
#             st.warning("⚠️ Could not load saved SHAP explainer. Will create on demand.")
#             models_dict['shap_explainer'] = None
                        
#     except Exception as e:
#         st.error(f"❌ Failed to load SHAP Explainer: {e}")
#         st.info("Will use feature importance for explanations")
#         models_dict['shap_explainer'] = None
    
#     try:
#         # Load Image Classification Model (Keras format)
#         st.write("Loading Image Classification Model...")
#         models_dict['image_model'] = load_model('best_model.keras')
#         st.success("✅ Image Classification model loaded")
#     except Exception as e:
#         st.error(f"❌ Failed to load Image Classification model: {e}")
#         models_dict['image_model'] = None
    
#     try:
#         # Load Segmentation Model (PyTorch format)
#         st.write("Loading Segmentation Model...")
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Load the state dict first to check its structure
#         checkpoint = torch.load('best_deeplabv3.pth', map_location=device)
        
#         # Check if aux_classifier keys exist
#         has_aux_classifier = any(key.startswith('aux_classifier') for key in checkpoint.keys())
        
#         if has_aux_classifier:
#             # Create model WITH aux_classifier since the saved model has it
#             seg_model = deeplabv3_resnet50(pretrained=False, aux_loss=True)
#         else:
#             # Create model WITHOUT aux_classifier
#             seg_model = deeplabv3_resnet50(pretrained=False, aux_loss=False)
        
#         # Modify the final layer for binary segmentation
#         seg_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        
#         # If model was trained with aux_classifier but we want to use without it,
#         # we can filter out those keys
#         if has_aux_classifier:
#             # Option 1: Load with aux_classifier
#             try:
#                 seg_model.load_state_dict(checkpoint, strict=True)
#                 st.success("✅ Segmentation model loaded with aux_classifier")
#             except:
#                 # Option 2: Filter out aux_classifier and load without it
#                 seg_model = deeplabv3_resnet50(pretrained=False, aux_loss=False)
#                 seg_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
                
#                 filtered_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('aux_classifier')}
#                 seg_model.load_state_dict(filtered_state_dict, strict=False)
#                 st.success("✅ Segmentation model loaded without aux_classifier")
#         else:
#             seg_model.load_state_dict(checkpoint, strict=True)
#             st.success("✅ Segmentation model loaded")
        
#         seg_model.to(device)
#         seg_model.eval()
#         models_dict['seg_model'] = seg_model
#         models_dict['device'] = device
        
#     except Exception as e:
#         st.error(f"❌ Failed to load Segmentation model: {e}")
#         st.info("Segmentation functionality will be disabled")
#         models_dict['seg_model'] = None
#         models_dict['device'] = torch.device("cpu")
    
#     return models_dict

# # Image preprocessing functions
# def preprocess_image_classification(image):
#     """Preprocess image for classification model"""
#     try:
#         image = image.convert('L')  # Convert to grayscale
#         image = image.resize((224, 224))
#         image_array = np.array(image) / 255.0
#         image_array = np.expand_dims(image_array, axis=0)
#         image_array = np.expand_dims(image_array, axis=-1)
#         return image_array
#     except Exception as e:
#         st.error(f"Error preprocessing image for classification: {e}")
#         return None

# def preprocess_image_segmentation(image, device):
#     """Preprocess image for segmentation model"""
#     try:
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485], std=[0.229])
#         ])
        
#         if image.mode != 'L':
#             image = image.convert('L')
        
#         image_tensor = transform(image).unsqueeze(0).to(device)
#         return image_tensor
#     except Exception as e:
#         st.error(f"Error preprocessing image for segmentation: {e}")
#         return None

# def encode_input_data(data, label_encoders):
#     """Encode input data using the same mappings as training"""
#     input_data = pd.DataFrame([data])
    
#     if label_encoders is not None:
#         # Use the actual label encoders from training
#         categorical_columns = ['sex', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
#         for col in categorical_columns:
#             if col in label_encoders:
#                 le = label_encoders[col]
#                 # Handle unknown categories by using the most frequent class
#                 if data[col] in le.classes_:
#                     input_data[col] = le.transform([data[col]])[0]
#                 else:
#                     # Use mode or most frequent class
#                     input_data[col] = 0  # Default fallback
#     else:
#         # Manual encoding as fallback - based on your training output
#         sex_mapping = {'Male': 1, 'Female': 0, 'Other': 2}
#         input_data['sex'] = sex_mapping.get(data['sex'], 0)
        
#         married_mapping = {'Yes': 1, 'No': 0}
#         input_data['ever_married'] = married_mapping.get(data['ever_married'], 0)
        
#         work_mapping = {
#             'Never_worked': 0, 
#             'children': 1, 
#             'Govt_job': 2, 
#             'Private': 3, 
#             'Self-employed': 4
#         }
#         input_data['work_type'] = work_mapping.get(data['work_type'], 3)
        
#         residence_mapping = {'Rural': 0, 'Urban': 1}
#         input_data['Residence_type'] = residence_mapping.get(data['Residence_type'], 1)
        
#         smoking_mapping = {
#             'never smoked': 0,
#             'formerly smoked': 1, 
#             'smokes': 1
#         }
#         input_data['smoking_status'] = smoking_mapping.get(data['smoking_status'], 0)
    
#     return input_data

# def predict_clinical_data(data, rf_model, scaler, label_encoders):
#     """Predict stroke from clinical data"""
#     try:
#         # Encode categorical data
#         input_data = encode_input_data(data, label_encoders)
        
#         # Scale numerical features
#         num_cols = ['age', 'avg_glucose_level', 'bmi']
#         input_data[num_cols] = scaler.transform(input_data[num_cols])
        
#         # Reorder columns to match training data
#         expected_columns = ['sex', 'age', 'hypertension', 'heart_disease', 'ever_married',
#                           'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
#         input_data = input_data[expected_columns]
        
#         # Predict
#         prediction = rf_model.predict(input_data)[0]
#         probability = rf_model.predict_proba(input_data)[0]
        
#         return prediction, probability, input_data
#     except Exception as e:
#         st.error(f"Error in clinical prediction: {str(e)}")
#         return None, None, None

# def create_shap_explanation(input_data, rf_model, scaler, label_encoders):
#     """Create SHAP explanation for the prediction"""
#     try:
#         # Create SHAP explainer
#         explainer = shap.TreeExplainer(rf_model)
        
#         # Calculate SHAP values for the input
#         shap_values = explainer.shap_values(input_data)
        
#         # Handle binary classification (2 classes)
#         if isinstance(shap_values, list) and len(shap_values) == 2:
#             # Use the positive class (stroke) SHAP values
#             values_to_plot = shap_values[1][0]  # First sample, stroke class
#             expected_value = explainer.expected_value[1]
#         else:
#             # Single output case
#             values_to_plot = shap_values[0]
#             expected_value = explainer.expected_value
        
#         # Feature names for display
#         feature_names = ['Sex', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 
#                         'Work Type', 'Residence Type', 'Glucose Level', 'BMI', 'Smoking Status']
        
#         # Create visualization
#         fig, ax = plt.subplots(figsize=(10, 6))
        
#         # Sort features by absolute SHAP value
#         abs_values = np.abs(values_to_plot)
#         sorted_indices = np.argsort(abs_values)
        
#         # Plot horizontal bar chart
#         colors = ['red' if v < 0 else 'blue' for v in values_to_plot[sorted_indices]]
#         bars = ax.barh(range(len(sorted_indices)), values_to_plot[sorted_indices], color=colors, alpha=0.7)
        
#         # Customize plot
#         ax.set_yticks(range(len(sorted_indices)))
#         ax.set_yticklabels([feature_names[i] for i in sorted_indices])
#         ax.set_xlabel('SHAP Value (Impact on Model Output)')
#         ax.set_title(f'Feature Impact on Stroke Prediction\n(Expected Value: {expected_value:.3f})')
#         ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.8)
        
#         # Add value labels on bars
#         for i, bar in enumerate(bars):
#             width = bar.get_width()
#             ax.text(width + (0.001 if width >= 0 else -0.001), bar.get_y() + bar.get_height()/2, 
#                    f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
#         plt.tight_layout()
#         return fig, values_to_plot, feature_names
        
#     except Exception as e:
#         st.error(f"Error creating SHAP explanation: {str(e)}")
#         return None, None, None

# def predict_image_classification(image, model):
#     """Predict stroke type from CT scan"""
#     try:
#         if model is None:
#             st.warning("Image classification model not available")
#             return "Model Unavailable", 0.0, [0.33, 0.33, 0.34]
        
#         processed_image = preprocess_image_classification(image)
#         if processed_image is None:
#             return None, None, None
            
#         prediction = model.predict(processed_image, verbose=0)
#         predicted_class = np.argmax(prediction[0])
#         confidence = np.max(prediction[0])
        
#         # Map classes (adjust based on your model)
#         class_names = ['Hemorrhage', 'No Stroke', 'Ischemic']
        
#         return class_names[predicted_class], confidence, prediction[0]
#     except Exception as e:
#         st.error(f"Error in image classification: {str(e)}")
#         return None, None, None

# def segment_stroke_region(image, model, device):
#     """Generate segmentation mask for stroke region"""
#     try:
#         if model is None:
#             st.warning("Segmentation model not available")
#             return None
            
#         processed_image = preprocess_image_segmentation(image, device)
#         if processed_image is None:
#             return None
        
#         with torch.no_grad():
#             output = model(processed_image)
#             mask = torch.sigmoid(output['out']).cpu().numpy()[0, 0]
        
#         return mask
#     except Exception as e:
#         st.error(f"Error in segmentation: {str(e)}")
#         return None

# def main():
#     # App header
#     st.markdown("<h1>🧠 AI-Powered Stroke Detection System</h1>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #ffffff; margin-bottom: 2rem;'>Advanced Medical AI for Early Stroke Detection using Clinical Data and CT Scans</p>", unsafe_allow_html=True)
    
#     # Load models
#     with st.spinner("Loading AI models..."):
#         models_dict = load_models()
    
#     if models_dict is None:
#         st.error("Critical error: Could not load essential models.")
#         st.stop()
    
#     # Extract models from dictionary
#     rf_model = models_dict.get('rf_model')
#     scaler = models_dict.get('scaler')
#     label_encoders = models_dict.get('label_encoders')
#     shap_explainer = models_dict.get('shap_explainer')
#     image_model = models_dict.get('image_model')
#     seg_model = models_dict.get('seg_model')
#     device = models_dict.get('device', torch.device("cpu"))
    
#     # Show model status
#     with st.expander("📊 Model Status", expanded=False):
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.write("**Clinical Models:**")
#             st.write(f"Random Forest: {'✅' if rf_model else '❌'}")
#             st.write(f"Scaler: {'✅' if scaler else '❌'}")
#             st.write(f"Label Encoders: {'✅' if label_encoders else '❌'}")
#         with col2:
#             st.write("**Image Models:**")
#             st.write(f"Classification: {'✅' if image_model else '❌'}")
#             st.write(f"Segmentation: {'✅' if seg_model else '❌'}")
#         with col3:
#             st.write("**Explainability:**")
#             st.write(f"SHAP: {'✅' if shap_explainer else '❌'}")
#             st.write(f"Device: {device}")
    
#     # Sidebar
#     st.sidebar.markdown("## 📋 Patient Information")
#     st.sidebar.markdown("---")
    
#     # Clinical data input
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown('<div class="main-container">', unsafe_allow_html=True)
#         st.markdown("### 👤 Clinical Data Input")
        
#         age = st.number_input("Age", min_value=0, max_value=120, value=50)
#         sex = st.selectbox("Sex", ["Male", "Female"])
#         hypertension = st.selectbox("Hypertension", ["No", "Yes"])
#         heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
#         ever_married = st.selectbox("Ever Married", ["No", "Yes"])
#         work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
#         residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
#         avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=100.0)
#         bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
#         smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     with col2:
#         st.markdown('<div class="main-container">', unsafe_allow_html=True)
#         st.markdown("### 🔬 CT Scan Upload")
        
#         uploaded_file = st.file_uploader(
#             "Upload CT Scan Image",
#             type=['png', 'jpg', 'jpeg'],
#             help="Upload a CT scan image for analysis"
#         )
        
#         if uploaded_file is not None:
#             image = Image.open(uploaded_file)
#             st.image(image, caption="Uploaded CT Scan", use_column_width=True)
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Analysis button
#     if st.button("🔍 Analyze Patient Data", key="analyze_btn"):
#         # Prepare clinical data
#         clinical_data = {
#             'age': age,
#             'sex': sex,
#             'hypertension': 1 if hypertension == "Yes" else 0,
#             'heart_disease': 1 if heart_disease == "Yes" else 0,
#             'ever_married': ever_married,
#             'work_type': work_type,
#             'Residence_type': residence_type,
#             'avg_glucose_level': avg_glucose_level,
#             'bmi': bmi,
#             'smoking_status': smoking_status
#         }
        
#         # Clinical prediction
#         if rf_model and scaler:
#             clinical_pred, clinical_prob, processed_input = predict_clinical_data(clinical_data, rf_model, scaler, label_encoders)
#         else:
#             st.error("Clinical analysis not available - missing models")
#             clinical_pred, clinical_prob, processed_input = None, None, None
        
#         # Image prediction
#         image_pred, image_conf, image_probs = None, None, None
#         if uploaded_file is not None and image_model:
#             image_pred, image_conf, image_probs = predict_image_classification(image, image_model)
#         elif uploaded_file is None:
#             st.warning("Please upload a CT scan image for complete analysis")
        
#         # Results display
#         if clinical_pred is not None:
#             st.markdown("## 📊 Analysis Results")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#                 st.metric("Clinical Analysis", 
#                          "Stroke Risk" if clinical_pred == 1 else "Normal", 
#                          f"{clinical_prob[1]*100:.1f}% confidence")
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             with col2:
#                 if image_pred:
#                     st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#                     st.metric("CT Scan Analysis", 
#                              image_pred, 
#                              f"{image_conf*100:.1f}% confidence")
#                     st.markdown('</div>', unsafe_allow_html=True)
#                 else:
#                     st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#                     st.metric("CT Scan Analysis", "Not Available", "No image uploaded")
#                     st.markdown('</div>', unsafe_allow_html=True)
            
#             with col3:
#                 # Combined analysis
#                 if clinical_pred == 0 and (image_pred == "No Stroke" or image_pred is None):
#                     final_result = "Normal"
#                 elif clinical_pred == 1 and (image_pred == "No Stroke" or image_pred is None):
#                     final_result = "Clinical Risk - Monitor"
#                 elif image_pred and image_pred != "No Stroke":
#                     final_result = f"Stroke Detected - {image_pred}"
#                 else:
#                     final_result = "Clinical Analysis Only"
                
#                 st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#                 st.metric("Final Diagnosis", final_result, "Combined Analysis")
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             # SHAP Analysis
#             if processed_input is not None:
#                 st.markdown('<div class="main-container">', unsafe_allow_html=True)
#                 st.markdown("### 🎯 Clinical Data Explanation (SHAP)")
                
#                 try:
#                     # Create SHAP explanation
#                     shap_fig, shap_values, feature_names = create_shap_explanation(
#                         processed_input, rf_model, scaler, label_encoders
#                     )
                    
#                     if shap_fig is not None:
#                         st.pyplot(shap_fig)
                        
#                         # Add interpretation
#                         st.markdown("**🔍 Interpretation:**")
#                         st.markdown("- **Blue bars**: Features increasing stroke risk")
#                         st.markdown("- **Red bars**: Features decreasing stroke risk") 
#                         st.markdown("- **Longer bars**: Stronger impact on prediction")
                        
#                         # Show top contributing factors
#                         if shap_values is not None and feature_names is not None:
#                             abs_values = np.abs(shap_values)
#                             top_indices = np.argsort(abs_values)[::-1][:3]
                            
#                             st.markdown("**🏆 Top Contributing Factors:**")
#                             for i, idx in enumerate(top_indices, 1):
#                                 impact = "increases" if shap_values[idx] > 0 else "decreases"
#                                 st.markdown(f"{i}. **{feature_names[idx]}** {impact} stroke risk (impact: {shap_values[idx]:.3f})")
#                     else:
#                         # Fallback to feature importance
#                         st.info("Using feature importance as explanation:")
#                         feature_names = ['Sex', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 
#                                        'Work Type', 'Residence Type', 'Glucose Level', 'BMI', 'Smoking Status']
#                         importance = rf_model.feature_importances_
                        
#                         fig, ax = plt.subplots(figsize=(10, 6))
#                         indices = np.argsort(importance)[::-1]
#                         bars = plt.bar(range(len(importance)), importance[indices], color='lightcoral', alpha=0.7)
#                         plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
#                         plt.title('Feature Importance from Random Forest Model')
#                         plt.xlabel('Features')
#                         plt.ylabel('Importance')
                        
#                         # Add value labels on bars
#                         for i, bar in enumerate(bars):
#                             height = bar.get_height()
#                             ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
#                                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                        
#                         plt.tight_layout()
#                         st.pyplot(fig)
                        
#                 except Exception as e:
#                     st.error(f"Error in SHAP analysis: {str(e)}")
                    
#                     # Fallback to feature importance
#                     st.info("Using feature importance as explanation:")
#                     feature_names = ['Sex', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 
#                                    'Work Type', 'Residence Type', 'Glucose Level', 'BMI', 'Smoking Status']
#                     importance = rf_model.feature_importances_
                    
#                     fig, ax = plt.subplots(figsize=(10, 6))
#                     indices = np.argsort(importance)[::-1]
#                     bars = plt.bar(range(len(importance)), importance[indices], color='lightcoral', alpha=0.7)
#                     plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
#                     plt.title('Feature Importance from Random Forest Model')
#                     plt.xlabel('Features')
#                     plt.ylabel('Importance')
                    
#                     for i, bar in enumerate(bars):
#                         height = bar.get_height()
#                         ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
#                                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                    
#                     plt.tight_layout()
#                     st.pyplot(fig)
                
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             # Image Analysis Results
#             if image_probs is not None:
#                 st.markdown('<div class="main-container">', unsafe_allow_html=True)
#                 st.markdown("### 🖼️ Image Analysis Results")
                
#                 # Create probability chart
#                 fig = go.Figure(data=[
#                     go.Bar(x=['Hemorrhage', 'No Stroke', 'Ischemic'], 
#                            y=image_probs*100,
#                            marker_color=['#ff6b6b', '#4ecdc4', '#feca57'])
#                 ])
#                 fig.update_layout(
#                     title="CT Scan Classification Probabilities",
#                     yaxis_title="Confidence (%)",
#                     template="plotly_dark",
#                     paper_bgcolor='rgba(0,0,0,0)',
#                     plot_bgcolor='rgba(0,0,0,0)'
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
                
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             # Segmentation for stroke cases
#             if image_pred and image_pred != "No Stroke" and seg_model and uploaded_file:
#                 st.markdown('<div class="main-container">', unsafe_allow_html=True)
#                 st.markdown("### 🎯 Stroke Region Segmentation")
                
#                 mask = segment_stroke_region(image, seg_model, device)
                
#                 if mask is not None:
#                     col1, col2, col3 = st.columns(3)
                    
#                     with col1:
#                         st.markdown("**Original Image**")
#                         st.image(image, use_column_width=True)
                    
#                     with col2:
#                         st.markdown("**Segmentation Mask**")
#                         fig, ax = plt.subplots()
#                         ax.imshow(mask, cmap='hot')
#                         ax.axis('off')
#                         st.pyplot(fig)
                    
#                     with col3:
#                         st.markdown("**Overlay**")
#                         # Create overlay
#                         img_array = np.array(image.convert('RGB'))
#                         mask_colored = plt.cm.hot(mask)[:, :, :3]
#                         overlay = 0.7 * img_array/255 + 0.3 * mask_colored
                        
#                         fig, ax = plt.subplots()
#                         ax.imshow(overlay)
#                         ax.axis('off')
#                         st.pyplot(fig)
                
#                 st.markdown('</div>', unsafe_allow_html=True)
            
#             # Final recommendation
#             st.markdown('<div class="main-container">', unsafe_allow_html=True)
#             st.markdown("### 💡 Medical Recommendation")
            
#             if "Normal" in final_result:
#                 st.success("✅ No immediate stroke risk detected. Continue regular health monitoring.")
#             elif "Monitor" in final_result:
#                 st.warning("⚠️ Clinical factors suggest increased stroke risk. Recommend consultation with healthcare provider.")
#             elif "Stroke Detected" in final_result:
#                 st.error("🚨 Potential stroke detected. Immediate medical attention required!")
#             else:
#                 st.info("ℹ️ Analysis completed with available models. Upload CT scan for complete assessment.")
            
#             st.markdown('</div>', unsafe_allow_html=True)
    
#     # Footer
#     st.markdown("---")
#     st.markdown(
#         "<p style='text-align: center; color: #ffffff; opacity: 0.7;'>"
#         "🏥 This AI system is for screening purposes only. Always consult healthcare professionals for medical decisions."
#         "</p>", 
#         unsafe_allow_html=True
#     )

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import pickle
import joblib
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io
import base64
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page configuration
st.set_page_config(
    page_title="🧠 AI Stroke Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
<style>
    /* Main background and text */
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Custom containers */
    .main-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .result-container {
        background: linear-gradient(145deg, #e74c3c, #c0392b);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        color: white;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .normal-container {
        background: linear-gradient(145deg, #27ae60, #229954);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        color: white;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .warning-container {
        background: linear-gradient(145deg, #f39c12, #e67e22);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        color: white;
        text-align: center;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Headers */
    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-weight: 700;
    }
    
    h2, h3 {
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(145deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        background: linear-gradient(145deg, #2980b9, #21618c);
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        border: 2px dashed rgba(255, 255, 255, 0.4);
        padding: 20px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
    }
    
    /* Metrics */
    .metric-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.15);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 8px;
    }
    
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.15);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 8px;
    }
    
    /* Text visibility improvements */
    .stMarkdown, .stText {
        color: #ffffff !important;
    }
    
    /* Improve sidebar text visibility */
    .css-1d391kg .stMarkdown {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_models():
    """Load all models with proper error handling"""
    models_dict = {}
    
    try:
        # Load Random Forest model
        try:
            models_dict['rf_model'] = joblib.load('random_forest_model.joblib')
        except:
            models_dict['rf_model'] = joblib.load('random_forest_model.pkl')
    except Exception as e:
        st.error(f"❌ Failed to load Random Forest model: {e}")
        return None
    
    try:
        # Load Scaler
        try:
            models_dict['scaler'] = joblib.load('scaler.joblib')
        except:
            models_dict['scaler'] = joblib.load('scaler.pkl')
    except Exception as e:
        st.error(f"❌ Failed to load Scaler: {e}")
        return None
    
    try:
        # Load Label Encoders
        with open('label_encoders.pkl', 'rb') as f:
            models_dict['label_encoders'] = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load Label Encoders: {e}")
        models_dict['label_encoders'] = None
    
    try:
        # Load SHAP Explainer
        try:
            with open('random_forest_shap_explainer.pkl', 'rb') as f:
                models_dict['shap_explainer'] = pickle.load(f)
        except:
            models_dict['shap_explainer'] = None
    except Exception as e:
        models_dict['shap_explainer'] = None
    
    try:
        # Load Image Classification Model
        models_dict['image_model'] = load_model('best_model.keras')
    except Exception as e:
        st.error(f"❌ Failed to load Image Classification model: {e}")
        models_dict['image_model'] = None
    
    try:
        # Load Segmentation Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load('best_deeplabv3.pth', map_location=device)
        
        # Check if aux_classifier keys exist
        has_aux_classifier = any(key.startswith('aux_classifier') for key in checkpoint.keys())
        
        if has_aux_classifier:
            seg_model = deeplabv3_resnet50(pretrained=False, aux_loss=True)
        else:
            seg_model = deeplabv3_resnet50(pretrained=False, aux_loss=False)
        
        # Modify the final layer for binary segmentation
        seg_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        
        if has_aux_classifier:
            try:
                seg_model.load_state_dict(checkpoint, strict=True)
            except:
                seg_model = deeplabv3_resnet50(pretrained=False, aux_loss=False)
                seg_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
                filtered_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('aux_classifier')}
                seg_model.load_state_dict(filtered_state_dict, strict=False)
        else:
            seg_model.load_state_dict(checkpoint, strict=True)
        
        seg_model.to(device)
        seg_model.eval()
        models_dict['seg_model'] = seg_model
        models_dict['device'] = device
        
    except Exception as e:
        st.error(f"❌ Failed to load Segmentation model: {e}")
        models_dict['seg_model'] = None
        models_dict['device'] = torch.device("cpu")
    
    return models_dict

# Image preprocessing functions
def preprocess_image_classification(image):
    """Preprocess image for classification model"""
    try:
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=-1)
        return image_array
    except Exception as e:
        st.error(f"Error preprocessing image for classification: {e}")
        return None

def preprocess_image_segmentation(image, device):
    """Preprocess image for segmentation model"""
    try:
        # Convert grayscale to RGB by duplicating channels
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode == 'RGBA':
            image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        st.error(f"Error preprocessing image for segmentation: {e}")
        return None

def encode_input_data(data, label_encoders):
    """Encode input data using the same mappings as training"""
    input_data = pd.DataFrame([data])
    
    if label_encoders is not None:
        # Use the actual label encoders from training
        categorical_columns = ['sex', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        
        for col in categorical_columns:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unknown categories by using the most frequent class
                if data[col] in le.classes_:
                    input_data[col] = le.transform([data[col]])[0]
                else:
                    # Use mode or most frequent class
                    input_data[col] = 0  # Default fallback
    else:
        # Manual encoding as fallback - based on your training output
        sex_mapping = {'Male': 1, 'Female': 0, 'Other': 2}
        input_data['sex'] = sex_mapping.get(data['sex'], 0)
        
        married_mapping = {'Yes': 1, 'No': 0}
        input_data['ever_married'] = married_mapping.get(data['ever_married'], 0)
        
        work_mapping = {
            'Never_worked': 0, 
            'children': 1, 
            'Govt_job': 2, 
            'Private': 3, 
            'Self-employed': 4
        }
        input_data['work_type'] = work_mapping.get(data['work_type'], 3)
        
        residence_mapping = {'Rural': 0, 'Urban': 1}
        input_data['Residence_type'] = residence_mapping.get(data['Residence_type'], 1)
        
        smoking_mapping = {
            'never smoked': 0,
            'formerly smoked': 1, 
            'smokes': 1
        }
        input_data['smoking_status'] = smoking_mapping.get(data['smoking_status'], 0)
    
    return input_data

def predict_clinical_data(data, rf_model, scaler, label_encoders):
    """Predict stroke from clinical data"""
    try:
        # Encode categorical data
        input_data = encode_input_data(data, label_encoders)
        
        # Scale numerical features
        num_cols = ['age', 'avg_glucose_level', 'bmi']
        input_data[num_cols] = scaler.transform(input_data[num_cols])
        
        # Reorder columns to match training data
        expected_columns = ['sex', 'age', 'hypertension', 'heart_disease', 'ever_married',
                          'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        input_data = input_data[expected_columns]
        
        # Predict
        prediction = rf_model.predict(input_data)[0]
        probability = rf_model.predict_proba(input_data)[0]
        
        return prediction, probability, input_data
    except Exception as e:
        st.error(f"Error in clinical prediction: {str(e)}")
        return None, None, None

def create_shap_explanation(input_data, rf_model, scaler, label_encoders):
    """Create SHAP explanation for the prediction"""
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(rf_model)
        
        # Calculate SHAP values for the input
        shap_values = explainer.shap_values(input_data)
        
        # Handle binary classification (2 classes)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Use the positive class (stroke) SHAP values
            values_to_plot = shap_values[1][0]  # First sample, stroke class
            expected_value = explainer.expected_value[1]
        else:
            # Single output case
            values_to_plot = shap_values[0]
            expected_value = explainer.expected_value
        
        # Feature names for display
        feature_names = ['Sex', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 
                        'Work Type', 'Residence Type', 'Glucose Level', 'BMI', 'Smoking Status']
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort features by absolute SHAP value
        abs_values = np.abs(values_to_plot)
        sorted_indices = np.argsort(abs_values)
        
        # Plot horizontal bar chart
        colors = ['#e74c3c' if v < 0 else '#3498db' for v in values_to_plot[sorted_indices]]
        bars = ax.barh(range(len(sorted_indices)), values_to_plot[sorted_indices], color=colors, alpha=0.8)
        
        # Customize plot with better visibility
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        ax.set_yticks(range(len(sorted_indices)))
        ax.set_yticklabels([feature_names[i] for i in sorted_indices], color='white', fontsize=11)
        ax.set_xlabel('SHAP Value (Impact on Model Output)', color='white', fontsize=12)
        ax.set_title(f'Feature Impact on Stroke Prediction\n(Expected Value: {expected_value:.3f})', 
                    color='white', fontsize=14, pad=20)
        ax.axvline(x=0, color='white', linewidth=1.2, alpha=0.8)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + (0.001 if width >= 0 else -0.001), bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', 
                   fontsize=9, color='white', fontweight='bold')
        
        plt.tight_layout()
        return fig, values_to_plot, feature_names
        
    except Exception as e:
        st.error(f"Error creating SHAP explanation: {str(e)}")
        return None, None, None

def predict_image_classification(image, model):
    """Predict stroke type from CT scan"""
    try:
        if model is None:
            st.warning("Image classification model not available")
            return "Model Unavailable", 0.0, [0.33, 0.33, 0.34]
        
        processed_image = preprocess_image_classification(image)
        if processed_image is None:
            return None, None, None
            
        prediction = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Map classes (adjust based on your model)
        class_names = ['Hemorrhage', 'No Stroke', 'Ischemic']
        
        return class_names[predicted_class], confidence, prediction[0]
    except Exception as e:
        st.error(f"Error in image classification: {str(e)}")
        return None, None, None

def segment_stroke_region(image, model, device):
    """Generate segmentation mask for stroke region"""
    try:
        if model is None:
            st.warning("Segmentation model not available")
            return None
            
        processed_image = preprocess_image_segmentation(image, device)
        if processed_image is None:
            return None
        
        with torch.no_grad():
            output = model(processed_image)
            mask = torch.sigmoid(output['out']).cpu().numpy()[0, 0]
        
        return mask
    except Exception as e:
        st.error(f"Error in segmentation: {str(e)}")
        return None

def main():
    # App header
    st.markdown("<h1>🧠 AI-Powered Stroke Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #ffffff; margin-bottom: 2rem;'>Advanced Medical AI for Early Stroke Detection using Clinical Data and CT Scans</p>", unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models..."):
        models_dict = load_models()
    
    if models_dict is None:
        st.error("Critical error: Could not load essential models.")
        st.stop()
    
    # Extract models from dictionary
    rf_model = models_dict.get('rf_model')
    scaler = models_dict.get('scaler')
    label_encoders = models_dict.get('label_encoders')
    shap_explainer = models_dict.get('shap_explainer')
    image_model = models_dict.get('image_model')
    seg_model = models_dict.get('seg_model')
    device = models_dict.get('device', torch.device("cpu"))
    
    # Show model status
    with st.expander("📊 Model Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Clinical Models:**")
            st.write(f"Random Forest: {'✅' if rf_model else '❌'}")
            st.write(f"Scaler: {'✅' if scaler else '❌'}")
            st.write(f"Label Encoders: {'✅' if label_encoders else '❌'}")
        with col2:
            st.write("**Image Models:**")
            st.write(f"Classification: {'✅' if image_model else '❌'}")
            st.write(f"Segmentation: {'✅' if seg_model else '❌'}")
        with col3:
            st.write("**Explainability:**")
            st.write(f"SHAP: {'✅' if shap_explainer else '❌'}")
            st.write(f"Device: {device}")
    
    # Sidebar
    st.sidebar.markdown("## 📋 Patient Information")
    st.sidebar.markdown("---")
    
    # Clinical data input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("### 👤 Clinical Data Input")
        
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=100.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("### 🔬 CT Scan Upload")
        
        uploaded_file = st.file_uploader(
            "Upload CT Scan Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a CT scan image for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded CT Scan", use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    if st.button("🔍 Analyze Patient Data", key="analyze_btn"):
        # Prepare clinical data
        clinical_data = {
            'age': age,
            'sex': sex,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        
        # Clinical prediction
        if rf_model and scaler:
            clinical_pred, clinical_prob, processed_input = predict_clinical_data(clinical_data, rf_model, scaler, label_encoders)
        else:
            st.error("Clinical analysis not available - missing models")
            clinical_pred, clinical_prob, processed_input = None, None, None
        
        # Image prediction
        image_pred, image_conf, image_probs = None, None, None
        if uploaded_file is not None and image_model:
            image_pred, image_conf, image_probs = predict_image_classification(image, image_model)
        elif uploaded_file is None:
            st.warning("Please upload a CT scan image for complete analysis")
        
        # Results display
        if clinical_pred is not None:
            st.markdown("## 📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Clinical Analysis", 
                         "Stroke Risk" if clinical_pred == 1 else "Normal", 
                         f"{clinical_prob[1]*100:.1f}% confidence")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if image_pred:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("CT Scan Analysis", 
                             image_pred, 
                             f"{image_conf*100:.1f}% confidence")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("CT Scan Analysis", "Not Available", "No image uploaded")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                # Combined analysis
                if clinical_pred == 0 and (image_pred == "No Stroke" or image_pred is None):
                    final_result = "Normal"
                elif clinical_pred == 1 and (image_pred == "No Stroke" or image_pred is None):
                    final_result = "Clinical Risk - Monitor"
                elif image_pred and image_pred != "No Stroke":
                    final_result = f"Stroke Detected - {image_pred}"
                else:
                    final_result = "Clinical Analysis Only"
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Final Diagnosis", final_result, "Combined Analysis")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # SHAP Analysis
            if processed_input is not None:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown("### 🎯 Clinical Data Explanation (SHAP)")
                
                try:
                    # Create SHAP explanation
                    shap_fig, shap_values, feature_names = create_shap_explanation(
                        processed_input, rf_model, scaler, label_encoders
                    )
                    
                    if shap_fig is not None:
                        st.pyplot(shap_fig)
                        
                        # Add interpretation
                        st.markdown("**🔍 Interpretation:**")
                        st.markdown("- **Blue bars**: Features increasing stroke risk")
                        st.markdown("- **Red bars**: Features decreasing stroke risk") 
                        st.markdown("- **Longer bars**: Stronger impact on prediction")
                        
                        # Show top contributing factors
                        if shap_values is not None and feature_names is not None:
                            abs_values = np.abs(shap_values)
                            top_indices = np.argsort(abs_values)[::-1][:3]
                            
                            st.markdown("**🏆 Top Contributing Factors:**")
                            for i, idx in enumerate(top_indices, 1):
                                impact = "increases" if shap_values[idx] > 0 else "decreases"
                                st.markdown(f"{i}. **{feature_names[idx]}** {impact} stroke risk (impact: {shap_values[idx]:.3f})")
                    else:
                        # Fallback to feature importance with better colors
                        st.info("Using feature importance as explanation:")
                        feature_names = ['Sex', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 
                                       'Work Type', 'Residence Type', 'Glucose Level', 'BMI', 'Smoking Status']
                        importance = rf_model.feature_importances_
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        indices = np.argsort(importance)[::-1]
                        bars = plt.bar(range(len(importance)), importance[indices], 
                                     color='#3498db', alpha=0.8, edgecolor='white', linewidth=1)
                        
                        # Styling improvements
                        ax.set_facecolor('none')
                        fig.patch.set_facecolor('none')
                        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], 
                                 rotation=45, ha='right', color='white', fontsize=10)
                        plt.title('Feature Importance from Random Forest Model', 
                                color='white', fontsize=14, pad=20)
                        plt.xlabel('Features', color='white', fontsize=12)
                        plt.ylabel('Importance', color='white', fontsize=12)
                        ax.tick_params(colors='white')
                        ax.spines['bottom'].set_color('white')
                        ax.spines['left'].set_color('white')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        # Add value labels on bars
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                   f'{height:.3f}', ha='center', va='bottom', 
                                   fontsize=9, color='white', fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error in SHAP analysis: {str(e)}")
                    
                    # Fallback to feature importance
                    st.info("Using feature importance as explanation:")
                    feature_names = ['Sex', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 
                                    'Work Type', 'Residence Type', 'Glucose Level', 'BMI', 'Smoking Status']
                    importance = rf_model.feature_importances_
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    indices = np.argsort(importance)[::-1]
                    bars = plt.bar(range(len(importance)), importance[indices], 
                                    color='#3498db', alpha=0.8, edgecolor='white', linewidth=1)
                    
                    # Styling improvements
                    ax.set_facecolor('none')
                    fig.patch.set_facecolor('none')
                    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], 
                                rotation=45, ha='right', color='white', fontsize=10)
                    plt.title('Feature Importance from Random Forest Model', 
                            color='white', fontsize=14, pad=20)
                    plt.xlabel('Features', color='white', fontsize=12)
                    plt.ylabel('Importance', color='white', fontsize=12)
                    ax.tick_params(colors='white')
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Add value labels on bars
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                f'{height:.3f}', ha='center', va='bottom', 
                                fontsize=9, color='white', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Image Analysis Results
            if image_probs is not None:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown("### 🖼️ Image Analysis Results")
                
                # Create probability chart
                fig = go.Figure(data=[
                    go.Bar(x=['Hemorrhage', 'No Stroke', 'Ischemic'], 
                            y=image_probs*100,
                            marker_color=['#e74c3c', '#27ae60', '#f39c12'],
                            text=[f'{p:.1f}%' for p in image_probs*100],
                            textposition='auto')
                ])
                fig.update_layout(
                    title="CT Scan Classification Probabilities",
                    yaxis_title="Confidence (%)",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title_font=dict(size=16, color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Segmentation for stroke cases
            if image_pred and image_pred != "No Stroke" and seg_model and uploaded_file:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown("### 🎯 Stroke Region Segmentation")
                
                mask = segment_stroke_region(image, seg_model, device)
                
                if mask is not None:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Original Image**")
                        st.image(image, use_column_width=True)
                    
                    with col2:
                        st.markdown("**Segmentation Mask**")
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(mask, cmap='plasma', alpha=0.8)
                        ax.axis('off')
                        ax.set_title('Detected Stroke Region', color='white', fontsize=12, pad=10)
                        fig.patch.set_facecolor('none')
                        st.pyplot(fig)
                    
                    with col3:
                        st.markdown("**Overlay**")
                        # Create overlay with better colors
                        img_array = np.array(image.convert('RGB'))
                        
                        # Normalize mask to 0-1 range
                        mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                        
                        # Create colored mask using plasma colormap
                        mask_colored = plt.cm.plasma(mask_norm)[:, :, :3]
                        
                        # Create overlay with better blending
                        overlay = 0.6 * (img_array/255.0) + 0.4 * mask_colored
                        overlay = np.clip(overlay, 0, 1)
                        
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(overlay)
                        ax.axis('off')
                        ax.set_title('Stroke Region Overlay', color='white', fontsize=12, pad=10)
                        fig.patch.set_facecolor('none')
                        st.pyplot(fig)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Final recommendation
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown("### 💡 Medical Recommendation")
            
            if "Normal" in final_result:
                st.success("✅ No immediate stroke risk detected. Continue regular health monitoring.")
            elif "Monitor" in final_result:
                st.warning("⚠️ Clinical factors suggest increased stroke risk. Recommend consultation with healthcare provider.")
            elif "Stroke Detected" in final_result:
                st.error("🚨 Potential stroke detected. Immediate medical attention required!")
            else:
                st.info("ℹ️ Analysis completed with available models. Upload CT scan for complete assessment.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #ffffff; opacity: 0.7;'>"
        "🏥 This AI system is for screening purposes only. Always consult healthcare professionals for medical decisions."
        "</p>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()