# Generated by Django 3.2.15 on 2022-11-23 06:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('App', '0005_alter_usermodel_after_uicon_alter_usermodel_u_icon'),
    ]

    operations = [
        migrations.AlterField(
            model_name='usermodel',
            name='after_uicon',
            field=models.ImageField(default=None, upload_to='after_icons'),
        ),
        migrations.AlterField(
            model_name='usermodel',
            name='u_icon',
            field=models.ImageField(default=None, upload_to='icons'),
        ),
    ]
