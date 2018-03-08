"""numberrecognize URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import include, url
from django.contrib import admin
from app.views import index,predict,data_generator,upload_csv_data,upload_csv_data_post
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^index/$',index),
    url(r'^predict/$',predict),
    url(r'^data_generator/$',data_generator),
    url(r'^upload_csv_data/$',upload_csv_data),
    url(r'^upload_csv_data_post/$',upload_csv_data_post),

]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
