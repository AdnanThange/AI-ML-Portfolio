from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import HeartPrediction

@admin.register(HeartPrediction)
class HeartPredictionAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'age',
        'sex',
        'chest_pain_type',
        'resting_bp',
        'cholesterol',
        'fasting_bs',
        'resting_ecg',
        'max_hr',
        'exercise_angina',
        'oldpeak',
        'st_slope',
        'prediction',
        'created_at',
    )
    list_filter = ('sex', 'exercise_angina', 'prediction', 'st_slope', 'chest_pain_type')
    search_fields = ('age', 'sex', 'prediction', 'chest_pain_type', 'resting_ecg')
    ordering = ('-created_at',)
