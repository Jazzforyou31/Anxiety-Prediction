from django import forms
from .models import AnxietyPrediction

class AnxietyPredictionForm(forms.ModelForm):
    class Meta:
        model = AnxietyPrediction
        fields = [
            'stress_sleep_impact',
            'stress_impact_score',
            'sleep_stress_severity',
            'sleep_hours',
            'activity_stress_balance',
            'high_risk_factors',
            'therapy_sessions',
            'stress_sleep_composite',
            'health_risk_score',
        ]
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add labels and help text for better user experience
        self.fields['stress_sleep_impact'].help_text = "How much does stress impact your sleep (1-10)"
        self.fields['stress_impact_score'].help_text = "Overall stress impact score (0-100)"
        self.fields['sleep_stress_severity'].help_text = "Severity of sleep-stress relationship (1-10)"
        self.fields['sleep_hours'].help_text = "Average sleep hours per night"
        self.fields['activity_stress_balance'].help_text = "Balance between activity and stress (-5 to 5)"
        self.fields['high_risk_factors'].help_text = "Number of high risk factors"
        self.fields['therapy_sessions'].help_text = "Therapy sessions per month"
        self.fields['stress_sleep_composite'].help_text = "Composite score of stress and sleep (1-10)"
        self.fields['health_risk_score'].help_text = "Overall health risk score (0-100)" 