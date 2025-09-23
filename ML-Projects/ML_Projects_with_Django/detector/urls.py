from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),                             # Home page
    path('upload/', views.upload_image, name='upload_image'),      # Image upload & detection
    path('live/', views.live_page, name='live_page'),              # Live camera page
    path('live_feed/', views.live_camera_feed, name='live_feed'),  # Streaming MJPEG feed
]
