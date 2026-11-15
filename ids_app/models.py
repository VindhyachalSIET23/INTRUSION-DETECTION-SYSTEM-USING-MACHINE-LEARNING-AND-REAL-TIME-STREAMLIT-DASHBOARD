from django.db import models
from django.utils import timezone


class NetworkPacketLog(models.Model):
    timestamp        = models.DateTimeField(default=timezone.now)
    src_ip           = models.CharField(max_length=64, default='0.0.0.0')
    src_port         = models.IntegerField(default=0)
    dst_ip           = models.CharField(max_length=64, default='0.0.0.0')
    dst_port         = models.IntegerField(default=80)
    protocol         = models.CharField(max_length=16, default='tcp')
    service          = models.CharField(max_length=32, default='http')
    flag             = models.CharField(max_length=16, default='SF')
    src_bytes        = models.IntegerField(default=0)
    # prediction: 'normal' or 'attack'
    prediction       = models.CharField(max_length=16, default='normal')
    # attack_category: Normal | DoS | Probe | R2L | U2R
    attack_category  = models.CharField(max_length=32, default='Normal')
    confidence       = models.FloatField(default=0.0)
    model_used       = models.CharField(max_length=64, default='RandomForest')

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.src_ip} ({self.attack_category})"


class BlockedIP(models.Model):
    ip_address   = models.CharField(max_length=64, unique=True)
    block_reason = models.CharField(max_length=255, default='Detected intrusion')
    blocked_at   = models.DateTimeField(auto_now_add=True)
    is_active    = models.BooleanField(default=True)

    class Meta:
        ordering = ['-blocked_at']

    def __str__(self):
        return f"{self.ip_address} ({'active' if self.is_active else 'unblocked'})"
