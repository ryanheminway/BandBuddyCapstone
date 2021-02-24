from django.conf.urls import url
from bandbuddy import views
from django.urls import path

urlpatterns = [
    url('^$', views.HomePageView.as_view()),
    path('config/', views.update_genre)
]