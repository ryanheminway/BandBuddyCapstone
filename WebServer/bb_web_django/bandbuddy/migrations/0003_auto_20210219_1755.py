# Generated by Django 3.1.6 on 2021-02-19 17:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('bandbuddy', '0002_auto_20210216_1848'),
    ]

    operations = [
        migrations.CreateModel(
            name='House',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('house_name', models.CharField(default=0, max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Student',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default=0, max_length=50)),
                ('house', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='bandbuddy.house')),
            ],
        ),
        migrations.DeleteModel(
            name='MyModel',
        ),
    ]
