from django.shortcuts import render, redirect
from django.views import View
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.db.models import Avg
from .forms import AnxietyPredictionForm
from .models import AnxietyPrediction
import pickle
import pandas as pd
import numpy as np
import os

def load_model():
    """Load the pre-trained anxiety prediction model"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'Model and Data', 'anxiety_prediction_model.pkl')
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    feature_names = model_package['feature_names']
    scaler = model_package['scaler']
    
    return model, feature_names, scaler

def predict_anxiety(input_data):
    """Make an anxiety prediction using the loaded model"""
    model, feature_names, scaler = load_model()
    
    # Create a DataFrame with the correct column names
    # Map form field names to model feature names - using the exact case that was used during training
    field_mapping = {
        'stress_sleep_impact': 'Stress_Sleep_Impact',
        'stress_impact_score': 'Stress_Impact_Score',
        'sleep_stress_severity': 'Sleep_Stress_Severity',
        'sleep_hours': 'Sleep Hours',
        'activity_stress_balance': 'Activity_Stress_Balance',
        'high_risk_factors': 'High_Risk_Factors',
        'therapy_sessions': 'Therapy Sessions (per month)',
        'stress_sleep_composite': 'Stress_Sleep_Composite',
        'health_risk_score': 'Health_Risk_Score'
    }
    
    # Create a dictionary with the mapped column names
    data_dict = {}
    for form_field, model_field in field_mapping.items():
        data_dict[model_field] = input_data.get(form_field)
    
    # Convert to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Make sure the dataframe has columns in the exact same order as the model was trained on
    # This is crucial for scikit-learn models and scalers
    df = df[feature_names]
    
    # Scale the features
    X_scaled = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    return prediction

def home_view(request):
    """View for the home page"""
    return render(request, 'anxiety_app/home.html')

# Authentication Views
def login_view(request):
    """View for user login"""
    if request.user.is_authenticated:
        return redirect('home')
        
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('home')  # Changed from dashboard to home
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'anxiety_app/login.html')

def register_view(request):
    """View for user registration"""
    if request.user.is_authenticated:
        return redirect('home')
        
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Removed auto-login
            messages.success(request, f'Account successfully created for {user.username}! Please log in.')
            return redirect('login')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = UserCreationForm()
    
    return render(request, 'anxiety_app/register.html', {'form': form})

def logout_view(request):
    """View for user logout"""
    logout(request)
    return redirect('home')

# Main App Views
class PredictionFormView(LoginRequiredMixin, View):
    """View for displaying and processing the prediction form"""
    login_url = 'login'
    
    def get(self, request):
        form = AnxietyPredictionForm()
        return render(request, 'anxiety_app/prediction_form.html', {'form': form})
    
    def post(self, request):
        form = AnxietyPredictionForm(request.POST)
        if form.is_valid():
            # Save the form but don't commit to DB yet
            prediction_obj = form.save(commit=False)
            
            # Set the user
            prediction_obj.user = request.user
            
            # Get the form data
            form_data = form.cleaned_data
            
            # Make prediction
            predicted_anxiety = predict_anxiety(form_data)
            
            # Update model with prediction
            prediction_obj.predicted_anxiety = predicted_anxiety
            prediction_obj.save()
            
            # Redirect to results page
            return redirect('prediction_result', pk=prediction_obj.pk)
        
        return render(request, 'anxiety_app/prediction_form.html', {'form': form})

@login_required(login_url='login')
def prediction_result_view(request, pk):
    """View for displaying prediction results"""
    prediction = AnxietyPrediction.objects.get(pk=pk)
    
    # Check if the prediction belongs to the current user
    if prediction.user != request.user:
        messages.error(request, "You do not have permission to view this prediction.")
        return redirect('dashboard')
        
    return render(request, 'anxiety_app/prediction_result.html', {'prediction': prediction})

@login_required(login_url='login')
def dashboard_view(request):
    """View for the user dashboard"""
    # Get user's predictions
    predictions = AnxietyPrediction.objects.filter(user=request.user)
    
    context = {
        'predictions': predictions,
    }
    
    if predictions.exists():
        # Get the latest prediction
        latest_prediction = predictions.first()
        
        # Calculate average score
        average_score = predictions.aggregate(avg_score=Avg('predicted_anxiety'))['avg_score']
        
        # Calculate average score as percentage (assuming max is 10)
        average_score_percent = min(average_score * 10, 100)
        
        # Calculate trend (difference between last two predictions)
        if predictions.count() > 1:
            previous_prediction = predictions[1]
            trend = latest_prediction.predicted_anxiety - previous_prediction.predicted_anxiety
        else:
            trend = 0
            
        context.update({
            'latest_prediction': latest_prediction,
            'average_score': average_score,
            'average_score_percent': average_score_percent,
            'trend': trend
        })
    
    return render(request, 'anxiety_app/dashboard.html', context)
