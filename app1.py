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

# Enhanced CSS for MERN-like styling with better text visibility
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main background and text */
    .main {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 30%, #2C5364 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 30%, #2C5364 100%);
    }
    
    /* Custom containers with glassmorphism */
    .main-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 
            0 8px 32px 0 rgba(31, 38, 135, 0.37),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.1);
    }
    
    .result-container {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.9), rgba(192, 57, 43, 0.9));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 
            0 15px 35px rgba(231, 76, 60, 0.2),
            0 5px 15px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .normal-container {
        background: linear-gradient(135deg, rgba(39, 174, 96, 0.9), rgba(34, 153, 84, 0.9));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 
            0 15px 35px rgba(39, 174, 96, 0.2),
            0 5px 15px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .warning-container {
        background: linear-gradient(135deg, rgba(243, 156, 18, 0.9), rgba(230, 126, 34, 0.9));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 
            0 15px 35px rgba(243, 156, 18, 0.2),
            0 5px 15px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Headers with better contrast */
    h1 {
        color: #ffffff !important;
        text-align: center;
        font-size: 3.5rem;
        margin-bottom: 2rem;
        text-shadow: 
            2px 2px 4px rgba(0,0,0,0.8),
            0 0 20px rgba(255,255,255,0.1);
        font-weight: 700;
        background: linear-gradient(45deg, #ffffff, #e3f2fd);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h2, h3 {
        color: #ffffff !important;
        text-shadow: 
            1px 1px 2px rgba(0,0,0,0.8),
            0 0 10px rgba(255,255,255,0.1);
        font-weight: 600;
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 30px;
        padding: 15px 40px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 
            0 8px 25px rgba(102, 126, 234, 0.3),
            0 3px 10px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 15px 40px rgba(102, 126, 234, 0.4),
            0 8px 25px rgba(0, 0, 0, 0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* File uploader enhancement */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 2px dashed rgba(102, 126, 234, 0.5);
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: rgba(255, 255, 255, 0.12);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Enhanced metrics */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 
            0 8px 25px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 12px 35px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.2);
    }
    
    /* Input fields with better visibility */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        background: rgba(255, 255, 255, 0.2) !important;
        padding: 10px 15px !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        background: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Text visibility improvements */
    .stMarkdown, .stText, p, span, div {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }
    
    /* Metric values */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: #ffffff !important;
    }
    
    /* Sidebar text */
    .css-1d391kg .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border-radius: 10px;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Labels */
    label {
        color: #ffffff !important;
        font-weight: 500 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Select box options */
    option {
        background: #2C5364 !important;
        color: white !important;
    }
    
    /* File uploader text */
    .stFileUploader label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background: transparent !important;
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
            st.success("✅ Random Forest model loaded")
        except:
            models_dict['rf_model'] = joblib.load('random_forest_model.pkl')
            st.success("✅ Random Forest model loaded (pkl)")
    except Exception as e:
        st.error(f"❌ Failed to load Random Forest model: {e}")
        return None
    
    try:
        # Load Scaler
        try:
            models_dict['scaler'] = joblib.load('scaler.joblib')
            st.success("✅ Scaler loaded")
        except:
            models_dict['scaler'] = joblib.load('scaler.pkl')
            st.success("✅ Scaler loaded (pkl)")
    except Exception as e:
        st.error(f"❌ Failed to load Scaler: {e}")
        return None
    
    try:
        # Load Label Encoders
        with open('label_encoders.pkl', 'rb') as f:
            models_dict['label_encoders'] = pickle.load(f)
        st.success("✅ Label Encoders loaded")
    except Exception as e:
        st.error(f"❌ Failed to load Label Encoders: {e}")
        models_dict['label_encoders'] = None
    
    try:
        # Load SHAP Explainer
        try:
            with open('random_forest_shap_explainer.pkl', 'rb') as f:
                models_dict['shap_explainer'] = pickle.load(f)
            st.success("✅ SHAP Explainer loaded")
        except:
            models_dict['shap_explainer'] = None
            st.info("ℹ️ SHAP explainer will be created on demand")
    except Exception as e:
        models_dict['shap_explainer'] = None
    
    try:
        # Load Image Classification Model
        models_dict['image_model'] = load_model('best_model.keras')
        st.success("✅ Image Classification model loaded")
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
                st.success("✅ Segmentation model loaded with aux_classifier")
            except:
                seg_model = deeplabv3_resnet50(pretrained=False, aux_loss=False)
                seg_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
                filtered_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('aux_classifier')}
                seg_model.load_state_dict(filtered_state_dict, strict=False)
                st.success("✅ Segmentation model loaded without aux_classifier")
        else:
            seg_model.load_state_dict(checkpoint, strict=True)
            st.success("✅ Segmentation model loaded")
        
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
        
        # Create visualization with enhanced styling
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort features by absolute SHAP value
        abs_values = np.abs(values_to_plot)
        sorted_indices = np.argsort(abs_values)
        
        # Plot horizontal bar chart with gradient colors
        colors = ['#e74c3c' if v < 0 else '#3498db' for v in values_to_plot[sorted_indices]]
        bars = ax.barh(range(len(sorted_indices)), values_to_plot[sorted_indices], 
                      color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Enhanced styling
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        ax.set_yticks(range(len(sorted_indices)))
        ax.set_yticklabels([feature_names[i] for i in sorted_indices], 
                          color='white', fontsize=12, fontweight='500')
        ax.set_xlabel('SHAP Value (Impact on Model Output)', 
                     color='white', fontsize=14, fontweight='600')
        ax.set_title(f'Feature Impact on Stroke Prediction\n(Expected Value: {expected_value:.3f})', 
                    color='white', fontsize=16, fontweight='700', pad=25)
        ax.axvline(x=0, color='white', linewidth=2, alpha=0.9)
        ax.tick_params(colors='white', labelsize=11)
        
        # Remove spines except for left and bottom
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        # Add value labels on bars with better positioning
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x = width + (0.002 if width >= 0 else -0.002)
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', 
                   fontsize=10, color='white', fontweight='bold')
        
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

def create_overlay_image(image, mask):
    """Create overlay image with proper dimension handling"""
    try:
        # Convert image to RGB array
        img_array = np.array(image.convert('RGB'))
        
        # Resize mask to match image dimensions
        mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
        
        # Normalize mask to 0-1 range
        mask_norm = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)
        
        # Create colored mask using plasma colormap
        mask_colored = plt.cm.plasma(mask_norm)[:, :, :3]
        
        # Ensure dimensions match for broadcasting
        if img_array.shape[:2] != mask_colored.shape[:2]:
            mask_colored = cv2.resize(mask_colored, (img_array.shape[1], img_array.shape[0]))
        
        # Create overlay with proper blending
        overlay = 0.6 * (img_array / 255.0) + 0.4 * mask_colored
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    except Exception as e:
        st.error(f"Error creating overlay: {str(e)}")
        return None

def main():
    # App header with enhanced styling
    st.markdown("<h1>🧠 AI-Powered Stroke Detection System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 3rem;'>
        <p style='font-size: 1.3rem; color: #ffffff; text-shadow: 1px 1px 2px rgba(0,0,0,0.7); font-weight: 400; line-height: 1.6;'>
            Advanced Medical AI for Early Stroke Detection using Clinical Data and CT Scans
        </p>
        <p style='font-size: 1rem; color: rgba(255,255,255,0.8); text-shadow: 1px 1px 2px rgba(0,0,0,0.5); margin-top: 1rem;'>
            Powered by Machine Learning and Deep Learning Technologies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
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
    
    # Show model status in a more attractive way
    with st.expander("📊 System Status & Model Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🔬 Clinical Models**")
            st.write(f"Random Forest: {'✅ Loaded' if rf_model else '❌ Failed'}")
            st.write(f"Data Scaler: {'✅ Loaded' if scaler else '❌ Failed'}")
            st.write(f"Label Encoders: {'✅ Loaded' if label_encoders else '❌ Failed'}")
        with col2:
            st.markdown("**🖼️ Image Models**")
            st.write(f"Classification: {'✅ Loaded' if image_model else '❌ Failed'}")
            st.write(f"Segmentation: {'✅ Loaded' if seg_model else '❌ Failed'}")
            st.write(f"Processing Device: {device}")
        with col3:
            st.markdown("**🔍 Explainability**")
            st.write(f"SHAP Analysis: {'✅ Available' if shap_explainer else '⚠️ On-demand'}")
            st.write(f"Feature Importance: ✅ Available")
            st.write(f"Model Interpretability: ✅ Enabled")
    
    # Sidebar with enhanced styling
    st.sidebar.markdown("## 📋 Patient Information")
    st.sidebar.markdown("---")
    
    # Clinical data input with better organization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("### 👤 Clinical Data Input")
        
        # Personal Information
        st.markdown("**Personal Information**")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=50, help="Patient's age in years")
        sex = st.selectbox("Biological Sex", ["Male", "Female"], help="Biological sex of the patient")
        ever_married = st.selectbox("Marital Status", ["No", "Yes"], help="Has the patient ever been married?")
        
        # Medical History
        st.markdown("**Medical History**")
        hypertension = st.selectbox("Hypertension", ["No", "Yes"], help="Does the patient have hypertension?")
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"], help="Does the patient have heart disease?")
        
        # Lifestyle Factors
        st.markdown("**Lifestyle & Environment**")
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], 
                                help="Type of work or employment status")
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], 
                                    help="Type of area where patient lives")
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"], 
                                    help="Patient's smoking history")
        
        # Clinical Measurements
        st.markdown("**Clinical Measurements**")
        avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, value=100.0, 
                                          help="Average blood glucose level")
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=100.0, value=25.0, 
                            help="Body Mass Index calculated from height and weight")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown("### 🔬 CT Scan Upload")
        
        uploaded_file = st.file_uploader(
            "Upload CT Scan Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a brain CT scan image for AI analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded CT Scan", use_column_width=True)
            
            # Show image info
            st.markdown("**Image Information:**")
            st.write(f"- **Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"- **Mode:** {image.mode}")
            st.write(f"- **Format:** {image.format}")
        else:
            st.info("📤 Please upload a CT scan image for complete analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button with enhanced styling
    st.markdown("<br>", unsafe_allow_html=True)
    
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
        
        # Progress bar for analysis
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Clinical prediction
        status_text.text("🔄 Analyzing clinical data...")
        progress_bar.progress(25)
        
        if rf_model and scaler:
            clinical_pred, clinical_prob, processed_input = predict_clinical_data(clinical_data, rf_model, scaler, label_encoders)
        else:
            st.error("Clinical analysis not available - missing models")
            clinical_pred, clinical_prob, processed_input = None, None, None
        
        # Image prediction
        image_pred, image_conf, image_probs = None, None, None
        if uploaded_file is not None and image_model:
            status_text.text("🔄 Analyzing CT scan image...")
            progress_bar.progress(50)
            image_pred, image_conf, image_probs = predict_image_classification(image, image_model)
        elif uploaded_file is None:
            st.warning("⚠️ Please upload a CT scan image for complete analysis")
        
        progress_bar.progress(75)
        status_text.text("🔄 Generating results...")
        
        # Results display
        if clinical_pred is not None:
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## 📊 Analysis Results")
            
            # Results metrics with enhanced styling
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                risk_level = "High Risk" if clinical_pred == 1 else "Low Risk"
                confidence = clinical_prob[1] if clinical_pred == 1 else clinical_prob[0]
                st.metric("🩺 Clinical Analysis", 
                         risk_level, 
                         f"{confidence*100:.1f}% confidence")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if image_pred and image_pred != "Model Unavailable":
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("🖼️ CT Scan Analysis", 
                             image_pred, 
                             f"{image_conf*100:.1f}% confidence")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("🖼️ CT Scan Analysis", "Not Available", "No image uploaded")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                # Combined analysis
                if clinical_pred == 0 and (image_pred == "No Stroke" or image_pred is None):
                    final_result = "Normal"
                    result_color = "normal"
                elif clinical_pred == 1 and (image_pred == "No Stroke" or image_pred is None):
                    final_result = "Clinical Risk - Monitor"
                    result_color = "warning"
                elif image_pred and image_pred != "No Stroke" and image_pred != "Model Unavailable":
                    final_result = f"Stroke Detected - {image_pred}"
                    result_color = "result"
                else:
                    final_result = "Clinical Analysis Only"
                    result_color = "normal"
                
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("🎯 Final Diagnosis", final_result, "Combined Analysis")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # SHAP Analysis with enhanced visualization
            if processed_input is not None:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown("### 🎯 Clinical Data Explanation (SHAP Analysis)")
                
                try:
                    # Create SHAP explanation
                    shap_fig, shap_values, feature_names = create_shap_explanation(
                        processed_input, rf_model, scaler, label_encoders
                    )
                    
                    if shap_fig is not None:
                        st.pyplot(shap_fig, bbox_inches='tight', facecolor='none')
                        
                        # Enhanced interpretation
                        st.markdown("**🔍 How to Read This Chart:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("- 🔵 **Blue bars**: Features that **increase** stroke risk")
                            st.markdown("- 🔴 **Red bars**: Features that **decrease** stroke risk")
                        with col2:
                            st.markdown("- 📏 **Longer bars**: **Stronger** impact on prediction")
                            st.markdown("- 📊 **SHAP values**: Quantify feature contributions")
                        
                        # Show top contributing factors
                        if shap_values is not None and feature_names is not None:
                            abs_values = np.abs(shap_values)
                            top_indices = np.argsort(abs_values)[::-1][:3]
                            
                            st.markdown("**🏆 Top 3 Contributing Factors:**")
                            for i, idx in enumerate(top_indices, 1):
                                impact = "**increases**" if shap_values[idx] > 0 else "**decreases**"
                                impact_color = "🔴" if shap_values[idx] > 0 else "🟢"
                                st.markdown(f"{i}. {impact_color} **{feature_names[idx]}** {impact} stroke risk (impact: {shap_values[idx]:.3f})")
                    else:
                        # Enhanced fallback visualization
                        st.info("📊 Displaying feature importance analysis:")
                        feature_names = ['Sex', 'Age', 'Hypertension', 'Heart Disease', 'Ever Married', 
                                       'Work Type', 'Residence Type', 'Glucose Level', 'BMI', 'Smoking Status']
                        importance = rf_model.feature_importances_
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        indices = np.argsort(importance)[::-1]
                        
                        # Create gradient colors
                        colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
                        bars = plt.bar(range(len(importance)), importance[indices], 
                                     color=colors[indices], alpha=0.8, edgecolor='white', linewidth=1.5)
                        
                        # Enhanced styling
                        ax.set_facecolor('none')
                        fig.patch.set_facecolor('none')
                        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], 
                                 rotation=45, ha='right', color='white', fontsize=11, fontweight='500')
                        plt.title('Feature Importance Analysis', 
                                color='white', fontsize=16, fontweight='700', pad=25)
                        plt.xlabel('Clinical Features', color='white', fontsize=14, fontweight='600')
                        plt.ylabel('Importance Score', color='white', fontsize=14, fontweight='600')
                        ax.tick_params(colors='white', labelsize=11)
                        
                        # Style spines
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                        ax.spines['bottom'].set_color('white')
                        ax.spines['left'].set_color('white')
                        ax.spines['bottom'].set_visible(True)
                        ax.spines['left'].set_visible(True)
                        
                        # Add value labels
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                                   f'{height:.3f}', ha='center', va='bottom', 
                                   fontsize=10, color='white', fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig, bbox_inches='tight', facecolor='none')
                        
                except Exception as e:
                    st.error(f"Error in explanation analysis: {str(e)}")
                    st.info("💡 Feature importance analysis temporarily unavailable")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced Image Analysis Results
            if image_probs is not None:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown("### 🖼️ CT Scan Analysis Results")
                
                # Create enhanced probability chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Hemorrhagic Stroke', 'Normal Brain', 'Ischemic Stroke'], 
                        y=image_probs*100,
                        marker_color=['#e74c3c', '#27ae60', '#f39c12'],
                        text=[f'{p:.1f}%' for p in image_probs*100],
                        textposition='auto',
                        textfont=dict(size=14, color='white', family='Inter'),
                        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
                    )
                ])
                
                fig.update_layout(
                    title={
                        'text': "CT Scan Classification Probabilities",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18, 'color': 'white', 'family': 'Inter'}
                    },
                    yaxis_title="Confidence Percentage (%)",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', family='Inter'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                max_prob_idx = np.argmax(image_probs)
                class_names = ['Hemorrhagic Stroke', 'Normal Brain', 'Ischemic Stroke']
                st.markdown(f"**🎯 Primary Diagnosis:** {class_names[max_prob_idx]} ({image_probs[max_prob_idx]*100:.1f}% confidence)")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced Segmentation for stroke cases
            if image_pred and image_pred != "No Stroke" and image_pred != "Model Unavailable" and seg_model and uploaded_file:
                st.markdown('<div class="main-container">', unsafe_allow_html=True)
                st.markdown("### 🎯 Stroke Region Segmentation")
                
                with st.spinner("🔄 Generating segmentation mask..."):
                    mask = segment_stroke_region(image, seg_model, device)
                
                if mask is not None:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📷 Original CT Scan**")
                        st.image(image, use_column_width=True, caption="Input CT scan image")
                    
                    with col2:
                        st.markdown("**🔥 Detected Stroke Region**")
                        fig, ax = plt.subplots(figsize=(6, 6))
                        im = ax.imshow(mask, cmap='plasma', alpha=0.9)
                        ax.axis('off')
                        ax.set_title('AI-Detected Stroke Region', color='white', fontsize=14, pad=15, fontweight='600')
                        fig.patch.set_facecolor('none')
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('Stroke Probability', color='white', fontsize=11)
                        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
                        
                        st.pyplot(fig, bbox_inches='tight', facecolor='none')
                    
                    with col3:
                        st.markdown("**🎨 Overlay Visualization**")
                        
                        # Create overlay with fixed broadcasting
                        overlay = create_overlay_image(image, mask)
                        
                        if overlay is not None:
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(overlay)
                            ax.axis('off')
                            ax.set_title('Stroke Region Overlay', color='white', fontsize=14, pad=15, fontweight='600')
                            fig.patch.set_facecolor('none')
                            st.pyplot(fig, bbox_inches='tight', facecolor='none')
                        else:
                            st.error("Could not create overlay visualization")
                    
                    # Segmentation statistics
                    st.markdown("**📊 Segmentation Statistics:**")
                    stroke_area = np.sum(mask > 0.5)  # Pixels with >50% stroke probability
                    total_area = mask.shape[0] * mask.shape[1]
                    affected_percentage = (stroke_area / total_area) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🧠 Affected Area", f"{affected_percentage:.2f}%", "of brain region")
                    with col2:
                        st.metric("📏 Stroke Pixels", f"{stroke_area:,}", "detected pixels")
                    with col3:
                        st.metric("🎯 Max Confidence", f"{np.max(mask)*100:.1f}%", "peak detection")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced Final Recommendation
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            st.markdown("### 💡 Medical Recommendation & Next Steps")
            
            if "Normal" in final_result:
                st.markdown('<div class="normal-container">', unsafe_allow_html=True)
                st.markdown("#### ✅ Low Risk Assessment")
                st.markdown("**Current Status:** No immediate stroke risk detected based on clinical data and imaging analysis.")
                st.markdown("**Recommendations:**")
                st.markdown("- Continue regular health monitoring and annual check-ups")
                st.markdown("- Maintain healthy lifestyle habits (diet, exercise, no smoking)")
                st.markdown("- Monitor blood pressure and glucose levels regularly")
                st.markdown("- Stay aware of stroke warning signs (FAST: Face, Arms, Speech, Time)")
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif "Monitor" in final_result:
                st.markdown('<div class="warning-container">', unsafe_allow_html=True)
                st.markdown("#### ⚠️ Elevated Risk - Monitoring Required")
                st.markdown("**Current Status:** Clinical factors suggest increased stroke risk. Imaging shows no acute findings.")
                st.markdown("**Immediate Actions:**")
                st.markdown("- Schedule consultation with healthcare provider within 1-2 weeks")
                st.markdown("- Discuss risk factor modification strategies")
                st.markdown("- Consider additional cardiovascular screening")
                st.markdown("- Monitor symptoms closely and seek immediate care if new symptoms develop")
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif "Stroke Detected" in final_result:
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown("#### 🚨 URGENT: Potential Stroke Detected")
                st.markdown("**Current Status:** AI analysis suggests possible stroke findings on CT scan.")
                st.markdown("**IMMEDIATE ACTIONS REQUIRED:**")
                st.markdown("- **SEEK EMERGENCY MEDICAL ATTENTION IMMEDIATELY**")
                st.markdown("- Call emergency services (911) or go to nearest emergency room")
                st.markdown("- Bring this analysis report and CT scan images")
                st.markdown("- Note time of symptom onset for medical team")
                st.markdown("**Remember:** Time is critical in stroke treatment!")
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.info("ℹ️ Analysis completed with available data. For comprehensive assessment, please provide both clinical information and CT scan images.")
            
            # Additional information
            st.markdown("**📞 Emergency Contacts:**")
            st.markdown("- **Emergency Services:** 911 (US) / 999 (UK) / 112 (EU)")
            st.markdown("- **Stroke Hotline:** 1-888-4-STROKE (1-888-478-7653)")
            st.markdown("- **National Stroke Association:** stroke.org")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; margin: 20px 0;'>
        <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-bottom: 10px;'>
            🏥 <strong>Medical Disclaimer:</strong> This AI system is designed for screening and educational purposes only.
        </p>
        <p style='color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-bottom: 5px;'>
            Always consult qualified healthcare professionals for medical decisions and treatment.
        </p>
        <p style='color: rgba(255,255,255,0.6); font-size: 0.8rem;'>
            🔬 Powered by Machine Learning • 🧠 AI Stroke Detection • 💡 Medical Innovation
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()