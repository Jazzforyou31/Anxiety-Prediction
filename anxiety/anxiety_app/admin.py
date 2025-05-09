from django.contrib import admin
from .models import AnxietyPrediction

# Register your models here.
@admin.register(AnxietyPrediction)
class AnxietyPredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'predicted_anxiety', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('id', 'predicted_anxiety')
    readonly_fields = ('predicted_anxiety', 'created_at')
