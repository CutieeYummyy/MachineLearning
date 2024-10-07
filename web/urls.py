from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'), 
    path('lab1/', views.LAB1, name='lab1'),
    path('lab2_1/', views.LAB2_1, name='lab2_1'),
    path('lab2_2/', views.LAB2_2, name='lab2_2'),
    path('lab3_1/', views.LAB3_1, name='lab3_1'),
    path('lab3_2/', views.LAB3_2, name='lab3_2'),
    path('lab3_3/', views.LAB3_3, name='lab3_3'),
    path('lab4_1/', views.LAB4_1, name='lab4_1'),
    path('lab4_2/', views.LAB4_2, name='lab4_2'),
    path('lab4_3/', views.LAB4_3, name='lab4_3'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)