from django.urls import path
from . import views

urlpatterns = [
    # Home page
    path('', views.home_view, name='home'),
    
    # Authentication URLs
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    
    # Dashboard URL
    path('dashboard/', views.dashboard_view, name='dashboard'),
    
    # Prediction URLs
    path('predict/', views.PredictionFormView.as_view(), name='prediction_form'),
    path('predict/result/<int:pk>/', views.prediction_result_view, name='prediction_result'),
]
