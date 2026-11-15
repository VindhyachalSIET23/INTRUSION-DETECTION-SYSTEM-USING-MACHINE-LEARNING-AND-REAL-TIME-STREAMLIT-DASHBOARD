from django.contrib import admin
from .models import NetworkPacketLog, BlockedIP


@admin.register(NetworkPacketLog)
class NetworkPacketLogAdmin(admin.ModelAdmin):
    list_display  = ('timestamp', 'src_ip', 'protocol', 'service', 'flag',
                     'prediction', 'attack_category', 'confidence', 'model_used')
    list_filter   = ('prediction', 'attack_category', 'protocol')
    search_fields = ('src_ip', 'dst_ip', 'service')
    ordering      = ('-timestamp',)
    readonly_fields = ('timestamp',)


@admin.register(BlockedIP)
class BlockedIPAdmin(admin.ModelAdmin):
    list_display  = ('ip_address', 'block_reason', 'blocked_at', 'is_active')
    list_filter   = ('is_active',)
    search_fields = ('ip_address',)
    ordering      = ('-blocked_at',)
