# Generated by Django 5.1.4 on 2024-12-05 11:49

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Camera",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=50)),
                ("servicename", models.CharField(max_length=50)),
                ("ipaddr", models.GenericIPAddressField()),
                ("createdon", models.DateTimeField(default=django.utils.timezone.now)),
                ("isdeleted", models.BooleanField()),
            ],
        ),
    ]