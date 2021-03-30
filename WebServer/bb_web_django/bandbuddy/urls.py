from django.conf.urls import url
from bandbuddy import views
from django.urls import path

urlpatterns = [
    url('^$', views.HomePageView.as_view()),
    path('config/', views.update_genre),
    path('download_combined/', views.download_combined,name="download_combined"),
    path('download_drums/', views.download_drums,name="download_drums"),
    path('download_input/', views.download_input,name="download_input"),
]