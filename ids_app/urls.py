from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('live-monitor/', views.live_monitor_view, name='live_monitor'),
    path('api/sniff-packet/', views.api_sniff_packet, name='api_sniff_packet'),
    path('packet-tester/', views.packet_tester_view, name='packet_tester'),
    path('batch-analysis/', views.batch_analysis_view, name='batch_analysis'),
    path('models-eval/', views.models_eval_view, name='models_eval'),
    path('dataset/', views.dataset_view, name='dataset'),
    path('firewall/', views.firewall_view, name='firewall'),
    path('attack-logs/', views.attack_logs_view, name='attack_logs'),
    path('export-logs-csv/', views.export_logs_csv, name='export_logs_csv'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
]
