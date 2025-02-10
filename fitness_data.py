from googleapiclient.discovery import build
import datetime
import pandas as pd

class FitnessData:
    def __init__(self, credentials):
        self.service = build('fitness', 'v1', credentials=credentials)
    
    def get_steps_data(self, days=7):
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(days=days)
        
        body = {
            "aggregateBy": [{
                "dataTypeName": "com.google.step_count.delta",
                "dataSourceId": "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps"
            }],
            "bucketByTime": {"durationMillis": 86400000},  # 1 day
            "startTimeMillis": int(start_time.timestamp() * 1000),
            "endTimeMillis": int(end_time.timestamp() * 1000)
        }
        
        response = self.service.users().dataset().aggregate(userId="me", body=body).execute()
        
        data = []
        for bucket in response['bucket']:
            date = datetime.datetime.fromtimestamp(int(bucket['startTimeMillis']) / 1000).date()
            if bucket['dataset'][0]['point']:
                steps = bucket['dataset'][0]['point'][0]['value'][0]['intVal']
            else:
                steps = 0
            data.append({'date': date, 'steps': steps})
        
        return pd.DataFrame(data)
    
    def get_heart_rate_data(self, days=1):
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(days=days)
        
        body = {
            "aggregateBy": [{
                "dataTypeName": "com.google.heart_rate.bpm",
                "dataSourceId": "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"
            }],
            "bucketByTime": {"durationMillis": 60000},  # 1 minute for real-time data
            "startTimeMillis": int(start_time.timestamp() * 1000),
            "endTimeMillis": int(end_time.timestamp() * 1000)
        }
        
        response = self.service.users().dataset().aggregate(userId="me", body=body).execute()
        
        data = []
        for bucket in response['bucket']:
            timestamp = datetime.datetime.fromtimestamp(int(bucket['startTimeMillis']) / 1000)
            if bucket['dataset'][0]['point']:
                hr = bucket['dataset'][0]['point'][0]['value'][0]['fpVal']
                data.append({'timestamp': timestamp, 'heart_rate': hr})
        
        return pd.DataFrame(data)

    def get_blood_oxygen_data(self, days=1):
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(days=days)
        
        body = {
            "aggregateBy": [{
                "dataTypeName": "com.google.oxygen_saturation",
                "dataSourceId": "derived:com.google.oxygen_saturation:com.google.android.gms:merged"
            }],
            "bucketByTime": {"durationMillis": 60000},  # 1 minute for real-time data
            "startTimeMillis": int(start_time.timestamp() * 1000),
            "endTimeMillis": int(end_time.timestamp() * 1000)
        }
        
        response = self.service.users().dataset().aggregate(userId="me", body=body).execute()
        
        data = []
        for bucket in response['bucket']:
            timestamp = datetime.datetime.fromtimestamp(int(bucket['startTimeMillis']) / 1000)
            if bucket['dataset'][0]['point']:
                spo2 = bucket['dataset'][0]['point'][0]['value'][0]['fpVal']
                data.append({'timestamp': timestamp, 'spo2': spo2})
        
        return pd.DataFrame(data)

    def get_blood_pressure_data(self, days=1):
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(days=days)
        
        body = {
            "aggregateBy": [{
                "dataTypeName": "com.google.blood_pressure",
                "dataSourceId": "derived:com.google.blood_pressure:com.google.android.gms:merged"
            }],
            "bucketByTime": {"durationMillis": 60000},  # 1 minute for real-time data
            "startTimeMillis": int(start_time.timestamp() * 1000),
            "endTimeMillis": int(end_time.timestamp() * 1000)
        }
        
        response = self.service.users().dataset().aggregate(userId="me", body=body).execute()
        
        data = []
        for bucket in response['bucket']:
            timestamp = datetime.datetime.fromtimestamp(int(bucket['startTimeMillis']) / 1000)
            if bucket['dataset'][0]['point']:
                systolic = bucket['dataset'][0]['point'][0]['value'][0]['fpVal']
                diastolic = bucket['dataset'][0]['point'][0]['value'][1]['fpVal']
                data.append({
                    'timestamp': timestamp,
                    'systolic': systolic,
                    'diastolic': diastolic
                })
        
        return pd.DataFrame(data)