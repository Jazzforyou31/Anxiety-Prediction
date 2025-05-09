from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class AnxietyPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    stress_sleep_impact = models.FloatField()
    stress_impact_score = models.FloatField()
    sleep_stress_severity = models.FloatField()
    sleep_hours = models.FloatField()
    activity_stress_balance = models.FloatField()
    high_risk_factors = models.IntegerField()
    therapy_sessions = models.IntegerField()
    stress_sleep_composite = models.FloatField()
    health_risk_score = models.FloatField()
    predicted_anxiety = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.id} - Score: {self.predicted_anxiety}"

    class Meta:
        ordering = ['-created_at']
