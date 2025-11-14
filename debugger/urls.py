from django.urls import path
from . import views

app_name = 'debugger'

urlpatterns = [
    path('', views.index, name='index'),
    path('execute/', views.execute_code, name='execute_code'),
    path('execution/<int:execution_id>/', views.get_execution, name='get_execution'),
]
